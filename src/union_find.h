#ifndef _UNION_FIND_H_
#define _UNION_FIND_H_

#include <unordered_map>

namespace pupil {

// Union and find structure.
class UnionFind {
  public:
    UnionFind(): num_components_(0),
        table_(std::unordered_map<int, int> ()) {};
    bool Insert(int node);
    int Find(int node);
    int Union(int node_a, int node_b);

    int NumComponents() const { return num_components_;};
  private:
    int num_components_;
    std::unordered_map<int, int> table_;
};

}  // namespace pupil

#endif//_UNION_FIND_H_