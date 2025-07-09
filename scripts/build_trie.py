import os
import struct
from collections import namedtuple

MAX_WORD_LEN = 37

GpuState = namedtuple("GpuState", ["transition_start_idx", "num_transitions", "lemma_offset"])
GpuTransition = namedtuple("GpuTransition", ["c", "next_state"])

class TrieNode:
    def __init__(self):
        self.children = {}
        self.lemma_offset = -1
        self.transitions_start = -1

form_to_lemma = {}
lemmas_set = set()

with open("uk_lemmatizer_dataset.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        form, lemma = parts[0], parts[1]
        form_to_lemma[form] = lemma
        lemmas_set.add(lemma)

lemma_offsets = {}
lemma_buffer = bytearray()

for lemma in sorted(lemmas_set):
    offset = len(lemma_buffer)
    lemma_offsets[lemma] = offset
    encoded = lemma.encode("utf-8")[:MAX_WORD_LEN - 1]
    lemma_buffer.extend(encoded + b'\x00')

root = TrieNode()
nodes = [root]

def insert(word, lemma_offset):
    node = root
    for b in word.encode("utf-8"):
        if b not in node.children:
            new_node = TrieNode()
            node.children[b] = new_node
            nodes.append(new_node)
        node = node.children[b]
    node.lemma_offset = lemma_offset

for word, lemma in form_to_lemma.items():
    insert(word, lemma_offsets[lemma])

transitions = []
states = []
node_indices = {id(n): idx for idx, n in enumerate(nodes)}

for node in nodes:
    sorted_children = sorted(node.children.items())
    node.transitions_start = len(transitions)
    for b, child in sorted_children:
        transitions.append(GpuTransition(c=b, next_state=node_indices[id(child)]))
    states.append(GpuState(
        transition_start_idx=node.transitions_start,
        num_transitions=len(sorted_children),
        lemma_offset=node.lemma_offset
    ))

output_dir = "../resources"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "states.bin"), "wb") as f:
    for s in states:
        f.write(struct.pack("IIi", s.transition_start_idx, s.num_transitions, s.lemma_offset))

with open(os.path.join(output_dir, "transitions.bin"), "wb") as f:
    for t in transitions:
        f.write(struct.pack("B I", t.c, t.next_state))

with open(os.path.join(output_dir, "lemmas.bin"), "wb") as f:
    f.write(lemma_buffer)

print(f"Serialized {len(states)} states, {len(transitions)} transitions, {len(lemma_buffer)} bytes of lemma_buffer.")
