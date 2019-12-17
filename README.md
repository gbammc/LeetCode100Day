[1014. Best Sightseeing Pair](https://leetcode.com/problems/best-sightseeing-pair/)
``` swift
```

[714. Best Time to Buy and Sell Stock with Transaction Fee](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)
``` swift
func maxProfit(_ prices: [Int], _ fee: Int) -> Int {
    guard prices.count > 1 else { return 0 }
    
    var buyStatus = [Int](repeating: 0, count: prices.count)
    var sellStatus = [Int](repeating: 0, count: prices.count)
    buyStatus[0] = -prices[0]
    for i in 1 ..< prices.count {
        buyStatus[i] = max(buyStatus[i - 1], sellStatus[i - 1] - prices[i])
        sellStatus[i] = max(sellStatus[i - 1], buyStatus[i - 1] + prices[i] - fee)
    }
    
    return sellStatus[prices.count - 1]
}
```

[1035. Uncrossed Lines](https://leetcode.com/problems/uncrossed-lines/)
``` swift
func maxUncrossedLines(_ A: [Int], _ B: [Int]) -> Int {
    var dp = [[Int]](repeating: [Int](repeating: 0, count: B.count + 1), count: A.count + 1)
    for i in 1 ... A.count {
        for j in 1 ... B.count {
            dp[i][j] = A[i - 1] == B[j - 1] ? dp[i - 1][j - 1] + 1 : max(dp[i][j - 1], dp[i - 1][j])
        }
    }
    return dp[A.count][B.count]
}
```

[1277. Count Square Submatrices with All Ones](https://leetcode.com/problems/count-square-submatrices-with-all-ones/submissions/)
``` swift
func countSquares(_ matrix: [[Int]]) -> Int {
    var res = 0
    // dp[i][j] means the biggest square with A[i][j] as bottom-right corner.
    var matrix = matrix
    for i in 0 ..< matrix.count {
        for j in 0 ..< matrix[0].count {
            if matrix[i][j] == 1 && i > 0 && j > 0 {
                matrix[i][j] += min(matrix[i - 1][j], min(matrix[i][j - 1], matrix[i - 1][j - 1]))
            }
            res += matrix[i][j]
        }
    }
    return res
}
```
[926. Flip String to Monotone Increasing](https://leetcode.com/problems/flip-string-to-monotone-increasing/)
``` swift
func minFlipsMonoIncr(_ S: String) -> Int {
    var f0 = 0
    var f1 = 0
    let c0 = Character("0").asciiValue!
    var index = S.startIndex
    while index != S.endIndex {
        let n = Int(S[index].asciiValue! - c0)
        f0 += n
        f1 = min(f0, f1 + 1 - n)
        index = S.index(after: index)
    }
    return f1
}
```

[900. RLE Iterator](https://leetcode.com/problems/rle-iterator/)
``` swift
class RLEIterator {
    
    var arr: [Int]
    var index = 0

    init(_ A: [Int]) {
        arr = A
    }
    
    func next(_ n: Int) -> Int {
        var n = n
        while index < arr.count && n > arr[index] {
            n -= arr[index]
            index += 2
        }
        
        if index >= arr.count {
            return -1
        }
        
        arr[index] -= n
        return arr[index + 1]
    }
}
```

[48. Rotate Image](https://leetcode.com/problems/rotate-image/)
```swift
func rotate(_ matrix: inout [[Int]]) {
    let count = matrix.count
    for i in 0 ..< count / 2 {
        (matrix[i], matrix[count - i - 1]) = (matrix[count - i - 1], matrix[i])
    }
    for i in 0 ..< count {
        for j in i + 1 ..< matrix[0].count {
            (matrix[i][j], matrix[j][i]) = (matrix[j][i], matrix[i][j])
        }
    }
}
```

[667. Beautiful Arrangement II](https://leetcode.com/problems/beautiful-arrangement-ii/)
``` swift
func constructArray(_ n: Int, _ k: Int) -> [Int] {
    var res = [Int]()
    var l = 1
    var r = k + 1
    while l <= r {
        res.append(l)
        l += 1
        if l <= r {
            res.append(r)
            r -= 1
        }
    }
    for i in (k + 2) ..< n + 1 {
        res.append(i)
    }
    return res
}
```

[1276. Number of Burgers with No Waste of Ingredients](https://leetcode.com/problems/number-of-burgers-with-no-waste-of-ingredients/)
``` swift
// 鸡兔同笼问题
func numOfBurgers(_ tomatoSlices: Int, _ cheeseSlices: Int) -> [Int] {
    let tmp = (tomatoSlices - 2 * cheeseSlices)
    let x = tmp / 2
    if tmp >= 0 && tmp % 2 == 0 && cheeseSlices - x >= 0 {
        return [x, cheeseSlices - x]
    }
    
    return []
}
```

[1275. Find Winner on a Tic Tac Toe Game](https://leetcode.com/contest/weekly-contest-165/problems/find-winner-on-a-tic-tac-toe-game/)
``` swift
func tictactoe(_ moves: [[Int]]) -> String {
    var board = [[Int]](repeating: [Int](repeating: -1, count: 3), count: 3)
    for i in 0 ..< moves.count {
        if i % 2 == 0 {
            board[moves[i][0]][moves[i][1]] = 1
        } else {
            board[moves[i][0]][moves[i][1]] = 0
        }
    }
    
    // checks rows and columns
    for i in 0 ..< 3 {
        if board[0][i] == board[1][i] && board[1][i] == board[2][i] && board[0][i] != -1 {
            return board[0][i] == 1 ? "A" : "B"
        }
        if board[i][0] == board[i][1] && board[i][1] == board[i][2] && board[i][0] != -1 {
            return board[i][0] == 1 ? "A" : "B"
        }
    }
    
    // checks diagonal
    if (board[0][0] == board[1][1] && board[1][1] == board[2][2] && board[0][0] != -1) {
        return board[0][0] == 1 ? "A" : "B"
    }
    if (board[0][2] == board[1][1] && board[2][0] == board[1][1] && board[0][2] != -1) {
        return board[0][2] == 1 ? "A" : "B"
    }
    
    if moves.count == 9 {
        return "Draw"
    }
    
    return "Pending"
}
```

[769. Max Chunks To Make Sorted](https://leetcode.com/problems/max-chunks-to-make-sorted/)
``` swift
func maxChunksToSorted(_ arr: [Int]) -> Int {
    var res = 1
    var curMax = arr[0]
    for i in 1 ..< arr.count {
        if curMax < arr[i] && curMax < i {
            res += 1
        }
        curMax = max(curMax, arr[i])
    }
    
    return res
}
```

[1052. Grumpy Bookstore Owner](https://leetcode.com/problems/grumpy-bookstore-owner/)
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