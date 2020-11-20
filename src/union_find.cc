#include "union_find.h"

#include <iostream>

namespace pupil {

bool UnionFind::Insert(int node) {
    if (table_.find(node) != table_.end()) {
        return false;
    }
    table_[node] = node;
    num_components_++;
    return true;
}

int UnionFind::Find(int node) {
    int root = node;
    while (root != table_[root]) {
		root = table_[root];
    }
    while (node != root) {
        int next = table_[node];
        table_[node] = root;
        node = next;
    }
    return root;
}

int UnionFind::Union(int node_a, int node_b) {
    int root_a = Find(node_a);
    int root_b = Find(node_b);
    if (root_a == root_b) { return root_a;}
    int root = (root_a < root_b) ? root_a : root_b;
    table_[root_a] = root;
    table_[root_b] = root;
    num_components_--;
    return root;
}

}  // namespace pupil