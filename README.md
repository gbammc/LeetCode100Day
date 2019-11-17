[1207. Unique Number of Occurrences](https://leetcode.com/problems/unique-number-of-occurrences/)
``` swift
func uniqueOccurrences(_ arr: [Int]) -> Bool {
    var dict = [Int: Int]()
    arr.forEach { dict[$0] = (dict[$0] ?? 0) + 1 } // `dict[$0] ?? 0` is more efficient than `dict[$0, default: 0]`
    
    return Set(dict.values).count == dict.values.count
}
```