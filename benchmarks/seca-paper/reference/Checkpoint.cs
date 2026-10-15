using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using SECA.SECA_DataClass;
using SourceCleanerEngine.DataClass;

public sealed class Input
{
    public double alpha = 0.7, beta = 0.5;
    public double alphaError = 0.1, betaError = 0.13, wordImportanceError = 0.3;
    public int branchMinimum = 1;
    public List<List<Document>> batches;
}
public sealed class Document
{
    public int id;
    public List<string> words;
}
public static class Program
{
    static readonly Dictionary<string, int> vocabulary = new Dictionary<string, int>();
    static readonly HashSet<int> sources = new HashSet<int>();
    static int row;
    static string outputDirectory;
    static void Save(object snapshot)
    {
        var cycle = (int)JObject.FromObject(snapshot)["cycle"];
        var path = Path.Combine(outputDirectory, "cycle-" + cycle.ToString("D4") + ".json");
        File.WriteAllText(path + ".tmp", JsonConvert.SerializeObject(new {
            schemaVersion = 1, mode = "accumulated-seca-serial-instrumented",
            vocabulary = vocabulary.Select(x => new { id = x.Value, word = x.Key }).ToArray(),
            snapshots = new[] { snapshot }
        }, Formatting.None));
        File.Move(path + ".tmp", path);
        Console.Error.WriteLine("Completed cycle " + cycle);
    }
    static JToken Capture(object value) { return value == null ? JValue.CreateNull() : JToken.FromObject(value); }
    static object Memberships(IEnumerable<KeyValuePair<int, HashSet<int>>> values)
    {
        return values.OrderBy(x => x.Key).Select(x => new { wordId = x.Key, sources = x.Value.OrderBy(y => y).ToArray() }).ToArray();
    }
    static object Topology(HKT h)
    {
        return new {
            id = h.HKTId, parentNodeId = h.parentNodeId,
            expectedWords = h.expectedWords_Dynamic.OrderBy(x => x.Key).Select(x => x.Value).ToArray(),
            nodes = h.nodes.OrderBy(n => n.sortingId).Select(n => new {
                id = n.nodeId, words = n.wordIds.OrderBy(w => w).ToArray(),
                sources = n.sourceIds.OrderBy(s => s).ToArray(),
                wordSources = Memberships(n.wordSourceIds),
                pendingSources = n.sourceIdsNewFromBatches.OrderBy(s => s).ToArray(),
                pendingWordSources = Memberships(n.wordSourceIdsNewFromBatches),
                expectedWords = n.expectedWords.OrderBy(x => x.Key).Select(x => x.Value).ToArray()
            }).ToArray()
        };
    }
    static object Tree(SECAAlgorithm algorithm)
    {
        return algorithm.hktList.Values.OrderBy(h => h.HKTId).Select(Topology).ToArray();
    }
    static Dictionary<int, SourceWord> Rows(List<Document> documents, int cycle)
    {
        if (documents == null || documents.Count == 0) throw new ArgumentException("Batches must be nonempty");
        var rows = new List<SourceWord>();
        foreach (var document in documents)
        {
            if (document.id <= 0 || !sources.Add(document.id)) throw new ArgumentException("Source IDs must be positive and globally unique");
            if (document.words == null || document.words.Count == 0 || document.words.Any(string.IsNullOrWhiteSpace) || document.words.Distinct().Count() != document.words.Count)
                throw new ArgumentException("Words must be nonempty, unique strings within each source");
            foreach (var word in document.words)
            {
                if (!vocabulary.ContainsKey(word)) vocabulary.Add(word, vocabulary.Count + 1);
                rows.Add(new SourceWord(++row, document.id, vocabulary[word], word, cycleId: cycle));
            }
        }
        var counts = rows.GroupBy(x => x.wordId).ToDictionary(x => x.Key, x => x.Count());
        foreach (var item in rows) item.wordNoOfSources = counts[item.wordId];
        // LINQ OrderBy is stable: source/word record order breaks frequency ties.
        return rows.OrderByDescending(x => x.wordNoOfSources).ToDictionary(x => x.sourceWordId);
    }
    public static int Main(string[] args)
    {
        try
        {
            if (args.Length != 2) throw new ArgumentException("Usage: checkpoint INPUT.json NEW_OUTPUT_DIRECTORY");
            var input = JsonConvert.DeserializeObject<Input>(File.ReadAllText(args[0]));
            if (input == null || input.batches == null || input.batches.Count == 0 || input.branchMinimum < 1 ||
                new[] {input.alpha, input.beta, input.alphaError, input.betaError, input.wordImportanceError}.Any(v => double.IsNaN(v) || double.IsInfinity(v) || v < 0 || v > 1))
                throw new ArgumentException("Invalid parameters or missing batches");
            outputDirectory = args[1];
            if (Directory.Exists(outputDirectory)) throw new ArgumentException("Output directory already exists");
            Directory.CreateDirectory(outputDirectory);
            Console.SetOut(Console.Error); // Keep reference logging away from JSON.
            var algorithm = new SECAAlgorithm(input.alpha, input.beta, minNoSourcesToCreateBranchForNode: input.branchMinimum);
            algorithm.buildTree(Rows(input.batches[0], 1), input.branchMinimum);
            Save(new { cycle = 1, tree = Capture(Tree(algorithm)) });
            for (int i = 1; i < input.batches.Count; i++)
            {
                var decisions = new List<JToken>();
                HarnessExecution.Observer = (h, deploy) => decisions.Add(Capture(new {
                    hktId = h.HKTId, accepted = h.isAccepted(deploy), triggered = !h.isAccepted(deploy),
                    topologyAtDecision = Topology(h), errors = h.hktChangeMetrics,
                    wordMetrics = h.wordChangeMetrics.OrderBy(x => x.Key).Select(x => new { wordId = x.Key, values = x.Value }).ToArray()
                }));
                // A reconstructed state-1 scope becomes state 0 for the next cycle.
                foreach (var h in algorithm.hktList.Values) h.isState1 = false;
                var incoming = Rows(input.batches[i], i + 1);
                var dead = new ConcurrentBag<long>();
                var deadNodes = new ConcurrentBag<long>();
                algorithm.updateTree(true, incoming, alphaThreshold: input.alpha, betaThreshold: input.beta,
                    alphaOption1Threshold: input.alphaError, betaOption1Threshold: input.betaError,
                    wordImportanceOption1Threshold: input.wordImportanceError, deadHKTs: dead, deadNodes: deadNodes);
                algorithm.clearMemory(i + 1, 0, dead, deadNodes, false);
                algorithm.copyDataDictionary(incoming, algorithm.sourceWordList, cycleId: i + 1);
                Save(new { cycle = i + 1, decisions, removedHktIds = dead.OrderBy(x => x).ToArray(), tree = Capture(Tree(algorithm)) });
            }
            return 0;
        }
        catch (Exception ex) { Console.Error.WriteLine(ex); return 1; }
    }
}
