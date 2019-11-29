``` swift
func maxSatisfied(_ customers: [Int], _ grumpy: [Int], _ X: Int) -> Int {
    // best solution
//    var res = 0
//    var window = 0  // sliding window to record the number of unsatisfied customers for X minutes
//    var maxValue = 0
//    for i in 0 ..< customers.count {
//        if grumpy[i] == 0 {
//            res += customers[i]
//        } else {
//            window += customers[i]
//        }
//        if i >= X {
//            window -= customers[i - X] * grumpy[i - X]
//        }
//        maxValue = max(maxValue, window)
//    }
//    
//    return res + maxValue


    // origin solution
    var res = 0
    var sums = [Int](repeating: 0, count: customers.count)
    for i in 0 ..< customers.count {
        res += customers[i] * (1 - grumpy[i])
        
        if i == 0 && grumpy[i] == 1 {
            sums[i] = customers[0]
        } else if i > 0 {
            sums[i] = sums[i - 1] + (grumpy[i] == 0 ? 0 : customers[i])
        }
    }
    
    var maxDesatisfied = sums[X - 1]
    for i in X ..< customers.count {
        maxDesatisfied = max(maxDesatisfied, sums[i] - sums[i - X])
    }
    return res + maxDesatisfied
}
```

[835. Image Overlap](https://leetcode.com/problems/image-overlap/)
``` swift
func largestOverlap(_ A: [[Int]], _ B: [[Int]]) -> Int {
    let n = A.count
    var res = 0
    var table = [[Int]](repeating: [Int](repeating: 0, count: 2 * n), count: 2 * n) // record the largest overlap with all shifts
    
    for i in 0 ..< n {
        for j in 0 ..< n where A[i][j] == 1 {
            for k in 0 ..< n {
                for l in 0 ..< n where B[k][l] == 1 {
                    table[i - k + n][j - l + n] += 1
                }
            }
        }
    }
    
    for row in table {
        for cell in row {
            res = max(res, cell)
        }
    }
    
    return res
}
```

[1031. Maximum Sum of Two Non-Overlapping Subarrays](https://leetcode.com/problems/maximum-sum-of-two-non-overlapping-subarrays/)
``` swift
func maxSumTwoNoOverlap(_ A: [Int], _ L: Int, _ M: Int) -> Int {
//        // best solution
//        var sums = [Int](repeating: A[0], count: A.count)
//        for i in 1 ..< A.count {
//            sums[i] = sums[i - 1] + A[i]
//        }
//        var res = sums[L + M - 1]
//        var Lmax = sums[L - 1] // max sum of contiguous L elements before the last M elements
//        var Mmax = sums[M - 1] // max sum of contiguous M elements before the last L elements
//        for i in L + M ..< A.count {
//            Lmax = max(Lmax, sums[i - M] - sums[i - M - L])
//            Mmax = max(Mmax, sums[i - L] - sums[i - M - L])
//            res = max(res, sums[i] - sums[i - L] + Mmax)
//            res = max(res, sums[i] - sums[i - M] + Lmax)
//        }
//        return res
    
    // origin solution
    var res = 0
    var sums = [Int](repeating: 0, count: A.count + 1)
    for i in 0 ..< A.count {
        sums[i + 1] = sums[i] + A[i]
    }
    for i in L ... A.count {
        for j in M ... A.count where j <= i - L || i <= j - M {
            let sumL = sums[i] - sums[i - L]
            let sumM = sums[j] - sums[j - M]
            res = max(res, sumL + sumM)
        }
    }
    return res
}
```

[969. Pancake Sorting](https://leetcode.com/problems/pancake-sorting/)
``` swift
func pancakeSort(_ A: [Int]) -> [Int] {
    var A = A
    var res = [Int]()
    
    func flip(_ array: inout [Int], _ index: Int) {
        var l = 0
        var r = index
        while l < r {
            (array[l], array[r]) = (array[r], array[l])
            l += 1
            r -= 1
        }
    }
    
    for i in stride(from: A.count, to: 1, by: -1) {
        var index = 0
        for j in 0 ..< i where A[j] == i {
            index = j
        }
        
        flip(&A, index)
        flip(&A, i - 1)
        
        res.append(index + 1)
        res.append(i)
    }
    return res
}
```

[1267. Count Servers that Communicate](https://leetcode.com/problems/count-servers-that-communicate/)
``` swift
func countServers(_ grid: [[Int]]) -> Int {
    var res = 0
    let m = grid.count
    let n = grid[0].count
    
    for i in 0 ..< m {
        for j in 0 ..< n where grid[i][j] == 1 {
            var found = false
            for k in 0 ..< m where grid[k][j] == 1 && k != i {
                found = true
                break
            }
            if !found {
                for k in 0 ..< n where grid[i][k] == 1 && k != j {
                    found = true
                    break
                }
            }
            if found {
                res += 1
            }
        }
    }
    
    return res
}
```

[1266. Minimum Time Visiting All Points](https://leetcode.com/contest/weekly-contest-164/problems/minimum-time-visiting-all-points/)
``` swift
func minTimeToVisitAllPoints(_ points: [[Int]]) -> Int {
    var res = 0
    for i in 1 ..< points.count {
        let diffX = abs(points[i][0] - points[i - 1][0])
        let diffY = abs(points[i][1] - points[i - 1][1])
        res += min(diffX, diffY) + abs(diffX - diffY)
    }
    return res
}
```

[1208. Get Equal Substrings Within Budget](https://leetcode.com/problems/get-equal-substrings-within-budget/)
``` swift
func equalSubstring(_ s: String, _ t: String, _ maxCost: Int) -> Int {
    let s = Array(s.utf8)
    let t = Array(t.utf8)
    
    var costs = [Int](repeating: 0, count: s.count)
    var rest = maxCost
    var start = 0
    var end = 0
    var res = 0
    
    for i in 0 ..< s.count {
        let cost = abs(Int(s[i]) - Int(t[i]))
        costs[i] = cost
        rest -= cost
        if rest < 0 {
            res = max(res, end - start)
            
            while start < end {
                rest += costs[start]
                start += 1
                if rest >= 0 {
                    break
                }
            }
            if rest < 0 {
                rest += cost
                start += 1
            }
        }
        end = i + 1
    }
    
    return max(res, end - start)
}
```

[1260. Shift 2D Grid](https://leetcode.com/contest/weekly-contest-163/problems/shift-2d-grid/)
```swift
func shiftGrid(_ grid: [[Int]], _ k: Int) -> [[Int]] {
    let n = grid.count
    let m = grid.first!.count
    let k = k % (n * m)
    var res = grid
    
    for i in 0 ..< n {
        for j in 0 ..< m {
            var targetJ = j + k
            let targetI = (i + targetJ / m) % n
            targetJ %= m
            res[targetI][targetJ] = grid[i][j]
        }
    }
    
    return res
}
```

[1253. Reconstruct a 2-Row Binary Matrix](https://leetcode.com/contest/weekly-contest-162/problems/reconstruct-a-2-row-binary-matrix/)
``` swift
func reconstructMatrix(_ upper: Int, _ lower: Int, _ colsum: [Int]) -> [[Int]] {
    var res = [[Int]](repeating: [Int](repeating: 0, count: colsum.count), count: 2)
    var upper = upper
    var lower = lower
    for i in 0 ..< colsum.count {
        if colsum[i] == 2 {
            upper -= 1
            lower -= 1
            res[0][i] = 1
            res[1][i] = 1
        } else if colsum[i] == 1 {
            if upper >= lower {
                upper -= 1
                res[0][i] = 1
            } else {
                lower -= 1
                res[1][i] = 1
            }
        }
        if upper < 0 || lower < 0 {
            return []
        }
    }
    if upper > 0 || lower > 0 {
        return []
    }
    return res
}
```

[1252. Cells with Odd Values in a Matrix](https://leetcode.com/contest/weekly-contest-162/problems/cells-with-odd-values-in-a-matrix/)
``` swift
func oddCells(_ n: Int, _ m: Int, _ indices: [[Int]]) -> Int {
    var res = 0
    var matrix = [[Int]](repeating: [Int](repeating: 0, count: m), count: n)
    for p in indices {
        let (row, col) = (p[0], p[1])
        for i in 0 ..< m {
            matrix[row][i] += 1
        }
        for i in 0 ..< n {
            matrix[i][col] += 1
        }
    }
    for i in 0 ..< n {
        for j in 0 ..< m where matrix[i][j] % 2 == 1 {
            res += 1
        }
    }
    return res
}
```

[1262. Greatest Sum Divisible by Three](https://leetcode.com/contest/weekly-contest-163/problems/greatest-sum-divisible-by-three/)
``` swift
func maxSumDivThree(_ nums: [Int]) -> Int {
    // dp is the maximum value after each iteration, with reminder 0, 2, 1
    var dp = [0, Int.min, Int.min]
    for n in nums {
        var nextdp = [0, 0, 0]
        nextdp[0] = max(dp[n % 3] + n, dp[0])
        nextdp[1] = max(dp[(n + 1) % 3] + n, dp[1])
        nextdp[2] = max(dp[(n + 2) % 3] + n, dp[2])
        dp = nextdp
    }
    return dp[0]
}
```

[1261. Find Elements in a Contaminated Binary Tree](https://leetcode.com/contest/weekly-contest-163/problems/find-elements-in-a-contaminated-binary-tree/)
``` swift
class FindElements {
    
    var values = Set<Int>()

    init(_ root: TreeNode?) {
        dfs(root, 0)
    }
    
    func dfs(_ root: TreeNode?, _ val: Int) {
        guard let root = root else { return }
        
        root.val = val
        values.insert(root.val)
        dfs(root.left, 2 * root.val + 1)
        dfs(root.right, 2 * root.val + 2)
    }
    
    func find(_ target: Int) -> Bool {
        return values.contains(target)
    }
    
}
```

[1207. Unique Number of Occurrences](https://leetcode.com/problems/unique-number-of-occurrences/)
``` swift
func uniqueOccurrences(_ arr: [Int]) -> Bool {
    var dict = [Int: Int]()
    
    // `dict[$0] ?? 0` is more efficient than `dict[$0, default: 0]`
    arr.forEach { dict[$0] = (dict[$0] ?? 0) + 1 }
    
    return Set(dict.values).count == dict.values.count
}
```