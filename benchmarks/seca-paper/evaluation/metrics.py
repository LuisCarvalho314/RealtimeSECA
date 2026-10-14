"""SECA equations 18–25 for complete hard partitions; no implicit conversion."""
from collections import Counter
from math import fsum, log

DEFINITIONS = {
    "version": "seca-hard-v1",
    "units": "fractions",
    "nmi": "2*MI/(H(predicted)+H(truth)); natural log; zero cells omitted",
    "bcubed_precision": "document mean of intersection/predicted cluster size",
    "bcubed_recall": "document mean of intersection/truth class size",
    "bcubed_f1": "document mean of per-document harmonic mean (not harmonic mean of means)",
    "purity": "sum of predicted-cluster majority counts / evaluated population",
    "empty_clusters": "ignored",
    "degenerate_nmi": "undefined (null) when both entropies are zero",
    "empty_population": "undefined (null)",
    "overlapping_or_unassigned": "scientific blocker; raw memberships preserved",
}

class ScientificBlocker(ValueError):
    pass


def unique_ids(values, context):
    if not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values):
        raise ValueError(f"{context}: expected nonempty string IDs")
    if len(values) != len(set(values)):
        raise ValueError(f"{context}: duplicate IDs")
    return set(values)


def validate_assignments(assignment, truth):
    if type(assignment.get("batch_index")) is not int or assignment["batch_index"] < 0:
        raise ValueError("invalid batch_index")
    population = unique_ids(assignment["evaluation_source_ids"], "population")
    labels = {}
    for record in truth:
        source = record["source_id"]
        unique_ids([source], "ground truth")
        if source in labels:
            raise ValueError("duplicate ground-truth source")
        if not isinstance(record["label"], str) or not record["label"]:
            raise ScientificBlocker("multiple/invalid ground-truth labels are not specified by the paper")
        labels[source] = record["label"]
    if population - labels.keys():
        raise ValueError("missing ground truth")
    memberships = {source: [] for source in population}
    cluster_ids = set()
    for cluster in assignment["clusters"]:
        cluster_id = cluster["cluster_id"]
        unique_ids([cluster_id], "cluster")
        if cluster_id in cluster_ids:
            raise ValueError("duplicate cluster ID")
        cluster_ids.add(cluster_id)
        sources = unique_ids(cluster["source_ids"], "cluster members")
        if sources - population:
            raise ValueError("unknown cluster document")
        for source in sources:
            memberships[source].append(cluster_id)
    unassigned = unique_ids(assignment["unassigned_source_ids"], "unassigned")
    if unassigned - population or unassigned != {s for s, c in memberships.items() if not c}:
        raise ValueError("inconsistent unassigned documents")
    return memberships, labels


def evaluate(assignment, truth):
    memberships, labels = validate_assignments(assignment, truth)
    if any(len(c) != 1 for c in memberships.values()):
        raise ScientificBlocker("overlapping/unassigned predictions: paper gives no conversion rule")
    n = len(memberships)
    predicted = {s: c[0] for s, c in memberships.items()}
    sizes = Counter(predicted.values())
    classes = Counter(labels[s] for s in predicted)
    cells = Counter((predicted[s], labels[s]) for s in predicted)
    result = {"evaluated_document_count": n, "predicted_cluster_count": len(sizes),
              "ground_truth_event_count": len(classes)}
    names = ("nmi", "bcubed_precision", "bcubed_recall", "bcubed_f1", "purity")
    if not n:
        return result | dict.fromkeys(names)
    entropy = lambda counts: -fsum((v/n)*log(v/n) for v in sorted(counts.values()))
    denominator = entropy(sizes) + entropy(classes)
    mi = fsum((v/n)*log(n*v/(sizes[c]*classes[t])) for (c, t), v in sorted(cells.items()))
    precisions, recalls, f1s = [], [], []
    for source in sorted(predicted):
        c, t = predicted[source], labels[source]
        p, r = cells[c, t]/sizes[c], cells[c, t]/classes[t]
        precisions.append(p)
        recalls.append(r)
        f1s.append(2*p*r/(p+r))
    majority = {}
    for (cluster, _), count in cells.items():
        majority[cluster] = max(majority.get(cluster, 0), count)
    return result | dict(zip(names, (2*mi/denominator if denominator else None,
        fsum(precisions)/n, fsum(recalls)/n, fsum(f1s)/n, sum(majority.values())/n)))
