"""
Edge-to-edge transition probabilities from SUMO .rou.xml.

**Edge flow grouping**
    Pass ``edge_groups`` (representative -> [member edges], same as ``EDGE_GROUPS``).
    Raw edges map to representatives before counting (e.g. D2, L2, L12 -> L12).

**Net-consistent normalization**
    With ``net_xml_path`` + ``edge_groups``, legal successors are defined between
    *groups* (any ``<connection>`` from a member of rep i to a member of rep j).

Use ``result.adjacency`` (symmetrized counts, normalized to [0,1]) for graph GP kernel weights.
``result.P`` (row-stochastic) is available for Markov-chain / diffusion interpretations.
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]
EdgeGroups = Mapping[str, Sequence[str]]



QUICKSTART_EDGE_GROUPS: Dict[str, List[str]] = {
    "L12": ["D2", "L2", "L12"],
    "L14": ["D4", "L4", "L14"],
    "L18": ["D5", "L5", "L18"],
    "L10": ["D7", "L7", "L10"],
    "L15": ["L15"],
    "L16": ["L16"],
}


def _ok_net_edge(eid: Optional[str], ignore_internal: bool) -> bool:
    if eid is None:
        return False
    if ignore_internal and str(eid).startswith(":"): #internal edges are edges that start with ":"
        return False
    return True


def legal_successor_mask_from_net_xml(
    net_xml_path: PathLike,
    edge_ids: Sequence[str],
    ignore_internal: bool = True,
) -> np.ndarray:
    edge_ids = list(edge_ids) # convert to list 
    idx = {e: i for i, e in enumerate(edge_ids)} # map each edge id to a unique integer index
    L = len(edge_ids) # number of edges
    mask = np.zeros((L, L), dtype=bool) # matrix of zeros with shape L×L
    root = ET.parse(Path(net_xml_path)).getroot()
    for conn in root.findall(".//connection"): # loop through all connections in the net.xml file
        f, t = conn.get("from"), conn.get("to") # get the from and to edges of the connection
        if not (_ok_net_edge(f, ignore_internal) and _ok_net_edge(t, ignore_internal)): # check if the from and to edges are valid
            continue
        if f not in idx or t not in idx or f == t:
            continue
        mask[idx[f], idx[t]] = True
    return mask


"""
“From rep i to rep j is legal if any member edge of rep i connects in net.xml to any member edge of rep j.
"""

def is_internal_edge(edge_id: Optional[str]) -> bool:
    return edge_id is not None and str(edge_id).startswith(":")


def filter_edge_sequence(
    edges: Sequence[str],
    ignore_internal: bool = True,
    allowed: Optional[set] = None,
) -> List[str]:
    out: List[str] = []
    for e in edges:
        e = str(e).strip()
        if not e or (ignore_internal and is_internal_edge(e)):
            continue
        if allowed is not None and e not in allowed:
            continue
        out.append(e)
    return out


def compress_consecutive(edges: Sequence[str]) -> List[str]:
    if not edges:
        return []
    out = [edges[0]]
    for e in edges[1:]:
        if e != out[-1]:
            out.append(e)
    return out


def member_to_representative_map(edge_groups: EdgeGroups) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for rep, members in edge_groups.items():
        for e in members:
            m[str(e).strip()] = str(rep).strip()
    return m


def map_sequence_to_representatives(
    edges: Sequence[str],
    edge_groups: EdgeGroups,
    ignore_internal: bool = True,
) -> List[str]:
    """Map members to reps; drop edges not in any group; compress duplicates."""
    m2r = member_to_representative_map(edge_groups)
    mapped: List[str] = []
    for e in edges:
        e = str(e).strip()
        if not e or (ignore_internal and is_internal_edge(e)):
            continue
        if e in m2r:
            mapped.append(m2r[e])
    return compress_consecutive(mapped)


def legal_successor_mask_for_representatives(
    net_xml_path: PathLike,
    representative_edge_ids: Sequence[str],
    edge_groups: EdgeGroups,
    ignore_internal: bool = True,
) -> np.ndarray:
    """True if any connection links a member of rep row to a member of rep col."""
    m2r = member_to_representative_map(edge_groups)
    reps = list(representative_edge_ids)
    idx = {e: i for i, e in enumerate(reps)}
    L = len(reps)
    mask = np.zeros((L, L), dtype=bool)
    root = ET.parse(Path(net_xml_path)).getroot()
    for conn in root.findall(".//connection"):
        f, t = conn.get("from"), conn.get("to")
        if not (_ok_net_edge(f, ignore_internal) and _ok_net_edge(t, ignore_internal)):
            continue
        rf, rt = m2r.get(f or ""), m2r.get(t or "")
        if rf is None or rt is None or rf == rt:
            continue
        if rf in idx and rt in idx:
            mask[idx[rf], idx[rt]] = True
    return mask


def parse_rou_xml_sequences( # parse the rou.xml file to get the sequences of edges
    rou_xml_path: PathLike,
    ignore_internal: bool = True,
) -> List[List[str]]:
    root = ET.parse(Path(rou_xml_path)).getroot()
    route_defs: Dict[str, str] = {}  # This builds a lookup table like: route_defs["r0"] = "L10 L7 D7
    for el in root.iter("route"):
        rid, es = el.get("id"), el.get("edges")
        if rid and es:
            route_defs[rid] = es

    sequences: List[List[str]] = [] # this will store the sequences of edges
    #If the vehicle says route="r0", it pulls the corresponding edges="..." string.
    for veh in root.iter("vehicle"):
        es = None
        r = veh.get("route")
        if r and r in route_defs:
            es = route_defs[r]
        if es is None:
            es = veh.get("edges")
        if es is None:
            for ch in veh:
                if ch.tag == "route" and ch.get("edges"):
                    es = ch.get("edges")
                    break
        if not es: 
            continue
        seq = compress_consecutive( # turn "L10 L7 D7" into ["L10", "L7", "D7"]
            filter_edge_sequence(es.replace(",", " ").split(), ignore_internal) # filter the edges to remove internal edges and duplicates
        )
        if seq:
            sequences.append(seq) # add the sequence to the list of sequences
    return sequences


def count_transitions_from_sequences( # this function counts the transitions from the sequences so from edge i to edge j
    sequences: Iterable[Sequence[str]],
    edge_ids: Optional[Sequence[str]] = None,
    ignore_internal: bool = True,
    edge_groups: Optional[EdgeGroups] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    seqs = [list(s) for s in sequences]
    if edge_groups is not None:
        seqs = [
            map_sequence_to_representatives(s, edge_groups, ignore_internal) for s in seqs
        ]
        seqs = [s for s in seqs if len(s) >= 1]

    if edge_ids is None:
        if edge_groups is not None:
            edge_ids = sorted(str(k).strip() for k in edge_groups.keys())
        else:
            seen: set = set()
            for s in seqs:
                for e in s:
                    e = str(e).strip()
                    if not e or (ignore_internal and is_internal_edge(e)):
                        continue
                    seen.add(e)
            edge_ids = sorted(seen)
    else:
        edge_ids = list(edge_ids)

    idx = {e: i for i, e in enumerate(edge_ids)}
    L = len(edge_ids)
    allowed = set(edge_ids)
    counts = np.zeros((L, L), dtype=np.int64)

    for s in seqs:
        if edge_groups is None:
            s = compress_consecutive(
                filter_edge_sequence(s, ignore_internal=ignore_internal, allowed=allowed)
            )
        else:
            s = compress_consecutive([e for e in s if e in allowed])
        for a, b in zip(s[:-1], s[1:]):
            if a in idx and b in idx and idx[a] != idx[b]:
                counts[idx[a], idx[b]] += 1

    outgoing = counts.sum(axis=1)
    return edge_ids, counts, outgoing


SinkPolicy = Literal["zeros", "uniform"]


def counts_to_probability_matrix( # normlize the counts to get the probability matrix
    counts: np.ndarray,
    laplace_alpha: float = 0.0,
    sink_policy: SinkPolicy = "zeros",
    legal_successors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    C = np.asarray(counts, dtype=np.float64)
    L = C.shape[0]
    row_sums = C.sum(axis=1)
    alpha = float(laplace_alpha)
    P = np.zeros_like(C)

    if legal_successors is None:
        if alpha > 0.0:
            warnings.warn(
                "Laplace smoothing without legal_successors spreads mass over all edges. "
                "Pass net_xml_path with edge_groups or legal_successor_mask_from_net_xml.",
                UserWarning,
                stacklevel=2,
            )
            Cs = C + alpha
            np.fill_diagonal(Cs, 0.0)  # counts diagonal is always 0; prevent self-loops after +alpha
            den = Cs.sum(axis=1, keepdims=True)
            den = np.where(den < 1e-15, 1.0, den)
            return Cs / den, row_sums
        for i in range(L):
            if row_sums[i] > 0:
                P[i, :] = C[i, :] / row_sums[i]
            elif sink_policy == "uniform":
                P[i, :] = 1.0 / L
        return P, row_sums

    legal = np.asarray(legal_successors, dtype=bool)
    for i in range(L):
        leg = legal[i]
        n_leg = int(leg.sum())
        if n_leg == 0:
            # True network sink: no legal successors in net.xml.
            # sink_policy is intentionally ignored here. we never invent
            # transitions that the road topology forbids.
            continue
        sum_leg = float(C[i, leg].sum())
        if alpha > 0.0:
            den = sum_leg + alpha * n_leg
            if den < 1e-15:
                continue
            P[i, leg] = (C[i, leg] + alpha) / den
        else:
            if sum_leg > 1e-15:
                P[i, leg] = C[i, leg] / sum_leg
            elif sink_policy == "uniform":
                P[i, leg] = 1.0 / n_leg
    return P, row_sums


def symmetrize_transition_matrix(
    P: np.ndarray,
    how: Literal["mean", "max", "sum"] = "mean",
) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    if how == "mean":
        return 0.5 * (P + P.T)
    if how == "max":
        return np.maximum(P, P.T)
    if how == "sum":
        return P + P.T
    raise ValueError(how)


@dataclass
class TransitionMatrixResult:
    edge_ids: List[str]
    counts: np.ndarray
    outgoing: np.ndarray
    P: np.ndarray
    adjacency: np.ndarray  # symmetrized counts normalized to [0, 1]; use this for graph GP kernels
    edges_no_outgoing: List[str] = field(default_factory=list)
    source: str = ""
    used_legal_successors_from_net: bool = False


def _finalize(
    sequences: List[List[str]],
    edge_ids: Optional[Sequence[str]],
    laplace_alpha: float,
    sink_policy: SinkPolicy,
    ignore_internal: bool,
    source: str,
    net_xml_path: Optional[PathLike] = None,
    edge_groups: Optional[EdgeGroups] = None,
) -> TransitionMatrixResult:
    eids, counts, outgoing = count_transitions_from_sequences(
        sequences,
        edge_ids=edge_ids,
        ignore_internal=ignore_internal,
        edge_groups=edge_groups,
    )
    legal = None
    used_net = False
    if net_xml_path is not None:
        if edge_groups is not None:
            legal = legal_successor_mask_for_representatives(
                net_xml_path, eids, edge_groups, ignore_internal=ignore_internal
            )
        else:
            legal = legal_successor_mask_from_net_xml(
                net_xml_path, eids, ignore_internal=ignore_internal
            )
        used_net = True

    P, _ = counts_to_probability_matrix(
        counts, laplace_alpha, sink_policy, legal_successors=legal
    )

    # Adjacency for graph GP: symmetrized raw counts, normalized to [0, 1].
    # Using counts (not P) preserves absolute connection strength and avoids
    # the degree-normalization distortion of the row-stochastic P matrix.
    W = symmetrize_transition_matrix(counts.astype(np.float64), how="mean")
    w_max = W.max()
    adjacency = W / w_max if w_max > 0 else W

    no_out = [eids[i] for i in range(len(eids)) if outgoing[i] == 0]
    return TransitionMatrixResult(
        edge_ids=eids,
        counts=counts,
        outgoing=outgoing,
        P=P,
        adjacency=adjacency,
        edges_no_outgoing=no_out,
        source=source,
        used_legal_successors_from_net=used_net,
    )


def transition_matrix_from_rou(
    rou_xml_path: PathLike,
    edge_ids: Optional[Sequence[str]] = None,
    laplace_alpha: float = 0.0,
    sink_policy: SinkPolicy = "zeros",
    ignore_internal: bool = True,
    net_xml_path: Optional[PathLike] = None,
    edge_groups: Optional[EdgeGroups] = None,
) -> TransitionMatrixResult:
    seqs = parse_rou_xml_sequences(rou_xml_path, ignore_internal=ignore_internal)
    return _finalize(
        seqs,
        edge_ids,
        laplace_alpha,
        sink_policy,
        ignore_internal,
        f"rou:{Path(rou_xml_path).name}",
        net_xml_path=net_xml_path,
        edge_groups=edge_groups,
    )

__all__ = [
    "QUICKSTART_EDGE_GROUPS",
    "TransitionMatrixResult",
    "transition_matrix_from_rou",
    "legal_successor_mask_from_net_xml",
    "legal_successor_mask_for_representatives",
    "member_to_representative_map",
    "map_sequence_to_representatives",
    "parse_rou_xml_sequences",
    "count_transitions_from_sequences",
    "counts_to_probability_matrix",
    "symmetrize_transition_matrix",
    "is_internal_edge",
    "filter_edge_sequence",
    "compress_consecutive",
]

