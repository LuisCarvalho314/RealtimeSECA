// Invokes the unmodified Release SECA.exe public algorithm API under Mono.
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using SECA.SECA_DataClass;
using SourceCleanerEngine.DataClass;

class OriginalAssemblyRunner
{
    static int Main(string[] args)
    {
        try { Run(args); return 0; }
        catch (Exception e) { Console.Error.WriteLine(e); return 1; }
    }

    static void Run(string[] args)
    {
        if (args.Length < 2) throw new ArgumentException("CONFIG BATCH...");
        var config = JObject.Parse(File.ReadAllText(args[0]));
        double alpha = (double)config["seca_thresholds"]["alpha"];
        double beta = (double)config["seca_thresholds"]["beta"];
        double ae = (double)config["seca_thresholds"]["alpha_error_threshold"];
        double be = (double)config["seca_thresholds"]["beta_error_threshold"];
        double wi = (double)config["seca_thresholds"]["word_importance_error_threshold"];
        int minimum = (int)config["hkt_builder"]["minimum_number_of_sources_to_create_branch_for_node"];
        var wordIds = new Dictionary<string, int>(StringComparer.Ordinal);
        var sourceIds = new Dictionary<string, int>(StringComparer.Ordinal);
        int rowId = 0;
        var batches = new List<Dictionary<int, SourceWord>>();
        foreach (var path in args.Skip(1))
        {
            var batch = JObject.Parse(File.ReadAllText(path));
            var rows = new List<SourceWord>();
            foreach (var source in (JArray)batch["sources"])
            {
                string sid = (string)source["source_id"];
                int sourceId;
                if (!sourceIds.TryGetValue(sid, out sourceId)) sourceIds.Add(sid, sourceId = sourceIds.Count + 1);
                var seen = new HashSet<string>(StringComparer.Ordinal);
                foreach (var token in (JArray)source["tokens"])
                {
                    string word = (string)token;
                    if (!seen.Add(word)) continue;
                    int wid;
                    if (!wordIds.TryGetValue(word, out wid)) wordIds.Add(word, wid = wordIds.Count + 1);
                    rows.Add(new SourceWord(++rowId, sourceId, wid, word));
                }
            }
            var frequencies = rows.GroupBy(r => r.wordId).ToDictionary(g => g.Key, g => g.Select(r => r.sourcedId).Distinct().Count());
            foreach (var row in rows) row.wordNoOfSources = frequencies[row.wordId];
            batches.Add(rows.OrderByDescending(r => r.wordNoOfSources).ToDictionary(r => r.sourceWordId, r => r));
        }
        var output = Console.Out;
        // Mono JIT compilation is process local. Exercise baseline and first update
        // on copies before the measured stream, then drop the temporary state.
        if (Environment.GetEnvironmentVariable("SECA_INTERNAL_WARMUP") == "1")
        {
            var warm = new SECAAlgorithm(alpha, beta, 1, minimum, false);
            Console.SetOut(TextWriter.Null);
            warm.buildTree(CloneRows(batches[0]), minimum);
            if (batches.Count > 1)
            {
                var deadH = new ConcurrentBag<long>();
                var deadN = new ConcurrentBag<long>();
                var next = CloneRows(batches[1]);
                warm.updateTree(true, next, alphaThreshold: alpha, betaThreshold: beta,
                    alphaOption1Threshold: ae, betaOption1Threshold: be,
                    wordImportanceOption1Threshold: wi, deadHKTs: deadH, deadNodes: deadN);
                warm.clearMemory(2, 3, deadH, deadN, false);
                warm.copyDataDictionary(next, warm.sourceWordList, cycleId: 2);
            }
            Console.SetOut(output);
            warm = null;
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }
        var streamWatch = Stopwatch.StartNew();
        var engine = new SECAAlgorithm(alpha, beta, 1, minimum, false);
        int processed = 0;
        for (int i = 0; i < batches.Count; i++)
        {
            var deadHkts = new ConcurrentBag<long>();
            var deadNodes = new ConcurrentBag<long>();
            var batch = batches[i];
            var currentProcess = Process.GetCurrentProcess();
            var cpuBefore = currentProcess.TotalProcessorTime;
            var sw = Stopwatch.StartNew();
            // Original implementation emits diagnostics inside hot loops. Discard to avoid terminal I/O.
            Console.SetOut(TextWriter.Null);
            if (i == 0) engine.buildTree(batch, minimum);
            else
            {
                engine.updateTree(true, batch, alphaThreshold: alpha, betaThreshold: beta,
                    alphaOption1Threshold: ae, betaOption1Threshold: be,
                    wordImportanceOption1Threshold: wi,
                    deadHKTs: deadHkts, deadNodes: deadNodes);
                engine.clearMemory(i + 1, 3, deadHkts, deadNodes, false);
                engine.copyDataDictionary(batch, engine.sourceWordList, cycleId: i + 1);
                foreach (var hkt in engine.hktList.Values) hkt.isState1 = false;
            }
            sw.Stop();
            currentProcess.Refresh();
            var cpuAfter = currentProcess.TotalProcessorTime;
            Console.SetOut(output);
            processed += batch.Values.Select(r => r.sourcedId).Distinct().Count();
            var live = engine.hktList.Values.ToArray();
            if (live.Count(h => h.parentNodeId == 0) != 1)
                throw new InvalidOperationException("original assembly did not retain one root HKT");
            var nodes = live.SelectMany(h => h.nodes).ToArray();
            var hByNode = nodes.ToDictionary(n => n.nodeId, n => n.HKTId);
            var parent = live.ToDictionary(h => h.HKTId, h => h.parentNodeId);
            int depth = 0;
            foreach (var h in live)
            {
                int d = 1; long p = h.parentNodeId;
                while (p != 0 && hByNode.ContainsKey(p) && parent.ContainsKey(hByNode[p]) && d <= live.Length)
                { d++; p = parent[hByNode[p]]; }
                depth = Math.Max(depth, d);
            }
            output.WriteLine(new JObject {
                ["engine"] = "csharp-original-assembly", ["batch"] = i,
                ["documents"] = processed, ["algorithm_seconds"] = sw.Elapsed.TotalSeconds,
                ["stream_wall_seconds"] = streamWatch.Elapsed.TotalSeconds,
                ["retained_documents"] = processed, ["hkts"] = live.Length,
                ["nodes"] = nodes.Length, ["refuge_nodes"] = nodes.Count(n => n.wordIds.Contains(-1)),
                ["max_depth"] = depth, ["source_memberships"] = nodes.Sum(n => n.sourceIds.Count),
                ["rss_after_batch_kib"] = currentProcess.PeakWorkingSet64 / 1024,
                ["algorithm_cpu_seconds"] = (cpuAfter - cpuBefore).TotalSeconds,
                ["removed_hkts"] = deadHkts.Count, ["reconstructed_hkts"] = JValue.CreateNull(),
                ["inspected_hkts"] = JValue.CreateNull()
            }.ToString(Newtonsoft.Json.Formatting.None));
        }
    }

    static Dictionary<int, SourceWord> CloneRows(Dictionary<int, SourceWord> rows)
    {
        return rows.ToDictionary(x => x.Key, x => new SourceWord(
            x.Value.sourceWordId, x.Value.sourcedId, x.Value.wordId,
            x.Value.word, x.Value.wordNoOfSources, x.Value.cycleId));
    }
}
