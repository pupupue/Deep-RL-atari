import numpy

"""   SumTree   """
class SumTree():
    def __init__(self, capacity):
        """
        :param capacity: Total capacity (Max number of Experiences)
        :return:
        """
        self.capacity = capacity
        self.mem_idx = 0
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.mem_idx + self.capacity - 1

        self.data[self.mem_idx] = data
        self.update(idx, p)

        """   rewrite with %   """
        self.mem_idx += 1
        if self.mem_idx >= self.capacity:
            self.mem_idx = 0

        """   to check when full   """
        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        # then propagate the change through tree
        # this method is faster than a recursive loop
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change
        # print(f"total: {self.total()}, capacity: {self.capacity}, fullity: {self.n_entries}")

    # get priority and sample
    def get_leaf(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


if __name__ == "__main__":
    numbers = [12, 7, 13, 14, 15, 16]
    sumtree = SumTree(6)
    val = ["a", "b", "c", "d", "e", "f"]
    p = 0
    print(sumtree.tree)

    for n in numbers:
        print(n, val[p])
        sumtree.add(n, val[p])
        p += 1
    print(sumtree.tree)
    print(sumtree.data, end="\n\n")
    sumtree.add(99, "x")
    print(sumtree.tree)
    print(sumtree.data, end="\n\n")
    sumtree.add(99, "x")
    print(sumtree.tree)
    print(sumtree.data, end="\n\n")
    # idx, val, data = sumtree.get_leaf(19)
    # print(idx, val, data)
    # sumtree.update(idx, 1)
    # print(sumtree.tree, end="\n\n")
    
    # idx, val, data = sumtree.get_leaf(19)
    # print(idx, val, data)
    # sumtree.update(idx, 1)
    # print(sumtree.tree, end="\n\n")

    
    # idx, val, data = sumtree.get_leaf(19)
    # print(idx, val, data)
    # sumtree.update(idx, 1)
    # print(sumtree.tree, end="\n\n")
