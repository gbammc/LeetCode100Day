[478. Generate Random Point in a Circle](https://leetcode.com/problems/generate-random-point-in-a-circle/)
``` swift
class Solution {

    var radius: Double = 0
    var x: Double = 0
    var y: Double = 0
    
    init(_ radius: Double, _ x_center: Double, _ y_center: Double) {
        self.radius = radius
        self.x = x_center
        self.y = y_center
    }
    
    func randPoint() -> [Double] {
        let t = Double.random(in: 0 ..< Double.pi * 2)
        // 用 sqrt 作为反函数
        let r = radius * sqrt(Double.random(in: 0 ..< 1))
        return [
            r * cos(t) + x,
            r * sin(t) + y,
        ]
    }

}
```


[322. Coin Change](https://leetcode.com/problems/coin-change/)
``` swift
// DP
// 时间复杂度：O(n * m)
// 空间复杂度：O(n)
func coinChange(_ coins: [Int], _ amount: Int) -> Int {      
    guard amount > 0 else { return 0 }
    var dp = [Int](repeating: 1000001, count: amount + 1)
    dp[0] = 0
    for i in 1 ... amount {
        for c in coins {
            if i >= c {
                dp[i] = min(dp[i], dp[i - c] + 1)
            }
        }
    }
    return dp.last! == 1000001 ? -1 : dp.last!
}

// DFS
func coinChange(_ coins: [Int], _ amount: Int) -> Int {
    var res = Int.max
    let coins = coins.sorted(by: <)
    
    func dfs(_ i: Int, _ n: Int, _ count: Int) {
        guard i >= 0 else { return }
        var times = n / coins[i]
        while times >= 0 {
            let left = n - times * coins[i]
            let newCount = count + times
            if left == 0 {
                res = min(res, newCount)
                break
            }
            if newCount + 1 >= res {
                break
            }
            dfs(i - 1, left, newCount)
            times -= 1
        }
    }
    dfs(coins.count - 1, amount, 0)
    return res == Int.max ? -1: res
}
```

[991. Broken Calculator](https://leetcode.com/problems/broken-calculator/)
``` swift
func brokenCalc(_ X: Int, _ Y: Int) -> Int {
    guard Y > X else { return X - Y }
    guard X != Y else { return 0 }
    var res = 0
    var queue = [Y]
    while !queue.isEmpty {
        var count = queue.count
        while count > 0 {
            let cur = queue.removeFirst()
            if cur < X {
                return res + X - cur
            }
            if cur == X {
                return res
            }
            if cur % 2 == 0 {
                queue.append(cur / 2)
            } else {
                queue.append(cur + 1)
            }
            count -= 1
        }
        res += 1
    }
    return -1
}
```

[524. Longest Word in Dictionary through Deleting](https://leetcode.com/problems/longest-word-in-dictionary-through-deleting/)
``` swift
// 时间复杂度：O(n * m)
// 空间复杂度：O(n * m)
func findLongestWord(_ s: String, _ d: [String]) -> String {
    let s = Array(s), d = d.sorted().map({ Array($0) }), sCount = s.count
    var result = [Character]()
    for str in d {
        let strCount = str.count
        var sIndex = 0, strIndex = 0
        while sIndex < sCount && strIndex < strCount {
            if s[sIndex] == str[strIndex] {
                strIndex += 1
            }
            sIndex += 1
        }
        if strIndex == strCount && result.count < str.count {
            result = str
        }
    }
    return String(result)
}
```

[895. Maximum Frequency Stack](https://leetcode.com/problems/maximum-frequency-stack/)
``` swift
class FreqStack {
    var maxFreq = 0
    var freqDict = [Int: [Int]]()
    var map = [Int: Int]()

     init() {
         
     }
     
     func push(_ x: Int) {
        map[x, default: 0] += 1
        let f = map[x]!
        if f > maxFreq {
            maxFreq += 1
        }
        freqDict[f, default: []].append(x)
     }
     
     func pop() -> Int {
        let res = freqDict[maxFreq]!.removeLast()
        map[res, default: 0] -= 1
        if freqDict[maxFreq]!.isEmpty {
            maxFreq -= 1
        }
        return res
     }
}
```

[645. Set Mismatch](https://leetcode.com/problems/set-mismatch/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func findErrorNums(_ nums: [Int]) -> [Int] {
    var tmp = nums
    var mis = 0
    var rep = 0
    for n in tmp {
        tmp[n - 1] *= -1
        if tmp[n - 1] > 0 {
            rep = n
        }
    }
    for (i, n) in tmp.enumerated() where n > 0 && i + 1 != rep {
        mis = i + 1
    }
    return [rep, mis]
}
```

[3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func lengthOfLongestSubstring(_ s: String) -> Int {
    guard s.count > 1 else { return s.count }
    let chars = Array(s)
    var dict = [Character: Int]()
    for c in "qwertyuiopasdfghjklzxcvbnm" {
        dict[c] = -1
    }
    var res = 0
    var ptr = 0
    for i in 0 ..< chars.count {
        ptr = max(ptr, dict[chars[i]]! + 1)
        res = max(res, i - ptr + 1)
        dict[chars[i]] = i
    }
    return res
}
```

[31. Next Permutation](https://leetcode.com/problems/next-permutation/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func nextPermutation(_ nums: inout [Int]) {
    guard nums.count > 1 else { return }
    
    var i = nums.count - 1
    while i - 1 >= 0 && nums[i - 1] >= nums[i] {
        i -= 1
    }
    
    if i == 0 {
        nums.sort()
        return
    }
    
    var j = i
    while j + 1 < nums.count && nums[j + 1] > nums[i - 1] {
        j += 1
    }
    
    (nums[i - 1], nums[j]) = (nums[j], nums[i - 1])
    let leftPart = nums[0 ... i - 1]
    let rightPart = nums[i...]
    let assemble = Array(leftPart + rightPart.sorted())
    nums = assemble
}
```

[1675. Minimize Deviation in Array](https://leetcode.com/problems/minimize-deviation-in-array/)
``` swift
// 时间复杂度：O(n * log(n))
// 空间复杂度：O(n)
func minimumDeviation(_ nums: [Int]) -> Int {
    guard !nums.isEmpty else { return 0 }

    var arr = [Int]()

    func insert(_ target: Int) {
        if arr.isEmpty {
            arr.append(target)
            return
        }

        var l = 0
        var r = arr.count - 1
        while l < r {
            let m = l + (r - l) / 2

            if arr[m] < target {
                l = m + 1
            } else {
                r = m
            }
        }

        if arr[l] < target {
            arr.insert(target, at: l + 1)
        } else if arr[l] > target {
            arr.insert(target, at: l)
        }
    }

    for n in nums {
        insert(n % 2 == 0 ? n : n * 2)
    }

    var res = arr.last! - arr.first!
    while arr.last! % 2 == 0 {
        insert(arr.last! / 2)
        arr.removeLast()
        res = min(res, arr.last! - arr.first!)
    }

    return res
}
```

[5. Longest Palindromic Substring](https://leetcode.com/submissions/detail/449549822/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func longestPalindrome(_ s: String) -> String {
    let chars = Array(s)
    let n = chars.count
    var l = 0
    var r = 0
    var len = 1
    var center = 0
    var start = 0
    
    while center < n {
        l = center
        r = center
        
        while r + 1 < n && chars[l]  == chars[r + 1] {
            r += 1
        }
        
        center = r + 1
        
        while l - 1 >= 0 && r + 1 < n && chars[l - 1] == chars[r + 1] {
            l -= 1
            r += 1
        }
        
        if len < r - l + 1 {
            len = r - l + 1
            start = l
        }
    }
    
    return String(chars[start ..< start + len])
}
```

[1680. Concatenation of Consecutive Binary Numbers](https://leetcode.com/problems/concatenation-of-consecutive-binary-numbers/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func concatenatedBinary(_ n: Int) -> Int {
    let mod = 1_000_000_007
    var res = 0
    var len = 0
    for i in 1 ... n {
        if i & (i - 1) == 0 {
            len += 1
        }
        
        res = (res << len) % mod
        res += i % mod
    }
    return res
}
```

[1631. Path With Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/)
``` swift
// 时间复杂度：O(n * n * log(n))
// 空间复杂度：O(n * n)
func minimumEffortPath(_ heights: [[Int]]) -> Int {
    let rc = heights.count, cc = rc > 0 ? heights[0].count : 0
    // BFS
    func helper(_ k: Int) -> Bool {
        var dp = [[Bool]](repeating: [Bool](repeating: true, count: cc), count: rc), nextQueue = [(Int, Int)]()
        func addToQueue(_ row: Int, _ col: Int) {
            dp[row][col] = false
            nextQueue.append((row, col))
        }
        addToQueue(0, 0)
        while !nextQueue.isEmpty {
            let queue = nextQueue
            nextQueue = []
            for (row, col) in queue {
                if row == rc - 1 && col == cc - 1 { 
                    return true 
                }
                if row > 0 && dp[row - 1][col] && abs(heights[row][col] - heights[row - 1][col]) <= k { 
                    addToQueue(row - 1, col) 
                }
                if col > 0 && dp[row][col - 1] && abs(heights[row][col] - heights[row][col - 1]) <= k { 
                    addToQueue(row, col - 1) 
                }
                if row < rc - 1 && dp[row + 1][col] && abs(heights[row][col] - heights[row + 1][col]) <= k { 
                    addToQueue(row + 1, col) 
                }
                if col < cc - 1 && dp[row][col + 1] && abs(heights[row][col] - heights[row][col + 1]) <= k { 
                    addToQueue(row, col + 1) 
                }
            }
        }
        return false
    }
    // 二分搜索
    var left = 0, right = heights.reduce(into: Int(0), { $0 = max($0, $1.max() ?? 0) })
    while left < right {
        let center = (left + right) / 2
        if helper(center) {
            right = center
        } else {
            left = center + 1
        }
    }
    return left
}
```

[1657. Determine if Two Strings Are Close](https://leetcode.com/problems/determine-if-two-strings-are-close/)
``` swift
// 时间复杂度：O(n + m)
// 空间复杂度：O(n + m)
func closeStrings(_ word1: String, _ word2: String) -> Bool {
    guard word1.count == word2.count else { return false }
    
    let w1 = Array(word1).reduce(into: [Character: Int]()) { $0[$1, default: 0] += 1 }
    let w2 = Array(word2).reduce(into: [Character: Int]()) { $0[$1, default: 0] += 1 }
    
    return Set(w1.keys) == Set(w2.keys) && Set(w1.values) == Set(w2.values)
}
```

[1673. Find the Most Competitive Subsequence](https://leetcode.com/problems/find-the-most-competitive-subsequence/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func mostCompetitive(_ nums: [Int], _ k: Int) -> [Int] {
    var res = [Int]()
    let n = nums.count
    for i in 0 ..< n {
        while res.count > 0 && n + res.count - i > k && res.last! > nums[i] {
            res.removeLast()
        }
        if res.count < k {
            res.append(nums[i])
        }
    }
    return res
}
```

[821. Shortest Distance to a Character](https://leetcode.com/problems/shortest-distance-to-a-character/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func shortestToChar(_ s: String, _ c: Character) -> [Int] {
    let chars = Array(s)
    var ret = [Int](repeating: 0, count: chars.count)
    var last = -1
    for (i, ch) in chars.enumerated() where ch == c {
        for j in stride(from: i - 1, to: last, by: -1) {
            ret[j] = min(i - j, j - (last != -1 ? last : -100000))
        }
        last = i
    }
    for j in 1 ..< chars.count - last {
        ret[last + j] = j
    }
    return ret
}
```

[138. Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func copyRandomList(_ head: Node?) -> Node? {
    let dummy = Node(-1)
    var cur = head
    while cur != nil {
        let copy = Node(cur!.val)
        copy.next = cur?.next
        cur?.next = copy
        cur = copy.next
    }
    cur = head
    while cur != nil {
        cur?.next?.random = cur?.random?.next
        cur = cur?.next?.next
    }
    cur = head
    var copy: Node? = dummy
    while cur != nil {
        copy?.next = cur?.next
        copy = copy?.next
        cur?.next = cur?.next?.next
        cur = cur?.next
    }
    return dummy.next
}
```

[1091. Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/)
``` swift
// 时间复杂度：O(n * m)
// 空间复杂度：O(n * m)
func shortestPathBinaryMatrix(_ grid: [[Int]]) -> Int {
    guard grid[0][0] == 0 else { return -1 }
    let n = grid.count
    let m = grid[0].count
    guard n > 1 && m > 1 else { return grid[0][0] == 0 ? 1 : -1 }
    var res = 1
    var visited = [[Bool]](repeating: [Bool](repeating: false, count: m), count: n)
    visited[0][0] = true
    var queue = [(0, 0)]
    let dir = [
        (1, 1), (1, -1), (-1, 1), (-1, -1),
        (1, 0), (0, 1), (0, -1), (-1, 0)
    ]
    
    while !queue.isEmpty {
        var q = [(Int, Int)]()
        for item in queue {
            for d in dir {
                let x = item.0 + d.0
                let y = item.1 + d.1
                if x >= 0 && x < n && y >= 0 && y < m && grid[x][y] == 0 && !visited[x][y] {
                    q.append((x, y))
                    visited[x][y] = true
                    
                    if x == n - 1 && y == m - 1 {
                        return res + 1
                    }
                }
            }
        }
        res += 1
        queue = q
    }
    
    return -1
}
```

[785. Is Graph Bipartite?](https://leetcode.com/problems/is-graph-bipartite/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func isBipartite(_ graph: [[Int]]) -> Bool {
    var memo = [Int](repeating: 0, count: graph.count)
    var set: Set<Int> = Array(0 ..< graph.count).reduce(into: Set<Int>()) { _ = $0.insert($1) }
    var queue = [0]
    while !set.isEmpty {
        let start = set.removeFirst()
        queue = [start]
        memo[start] = 1
        while !queue.isEmpty {
            let n = queue.removeLast()
            for e in graph[n] {
                if memo[n] == memo[e] {
                    return false
                }
                if memo[e] == 0 {
                    memo[e] = memo[n] * -1
                    queue.append(e)
                    set.remove(e)
                }
            }
        }
    }
    
    return true
}
```

[127. Word Ladder](https://leetcode.com/problems/word-ladder/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func ladderLength(_ beginWord: String, _ endWord: String, _ wordList: [String]) -> Int {
    let set = Set<String>(wordList)
    var queue = [String]()
    var visited = Set<String>()
    
    queue.append(beginWord)
    visited.insert(beginWord)
    
    var level = 0
    
    while !queue.isEmpty {
        var size = queue.count
        level += 1
        
        while size > 0 {
            size -= 1
            
            let word = queue.removeFirst()
            if word == endWord {
                return level
            }
            
            for i in 0 ..< word.count {
                var chars = Array(word)
                for c in "qwertyuiopasdfghjklzxcvbnm" {
                    chars[i] = c
                    let newWord = String(chars)
                    if set.contains(newWord) && !visited.contains(newWord) {
                        queue.append(newWord)
                        visited.insert(newWord)
                    }
                }
            }
        }
    }
    
    return 0
}
```

[3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func lengthOfLongestSubstring(_ s: String) -> Int {
    guard s.count > 1 else { return s.count }
    let chars = Array(s)
    var dict = [Character: Int]()
    var res = 0
    var ptr = 0
    for i in 0 ..< chars.count {
        if let last = dict[chars[i]], last >= ptr {
            ptr = last + 1
        }
        res = max(res, i - ptr + 1)
        dict[chars[i]] = i
    }
    return res
}
```

[880. Decoded String at Index](https://leetcode.com/problems/decoded-string-at-index/)
``` swift
// 时间复杂度：O(K)
// 空间复杂度：O(1)
func decodeAtIndex(_ S: String, _ K: Int) -> String {
    let S = Array(S)
    var i = 0
    var N = 0
    var K = K
    while N < K {
        N = S[i].isNumber ? N * Int(String(S[i]))! : N + 1
        i += 1
    }
    while i > 0 {
        i -= 1
        if S[i].isNumber {
            N /= Int(String(S[i]))!
            K %= N
        } else {
            if K % N == 0 {
                return String(S[i])
            }
            N -= 1
        }
    }

    return ""
}
```

[556. Next Greater Element III](https://leetcode.com/problems/next-greater-element-iii/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func nextGreaterElement(_ n: Int) -> Int {
    var digits = String(n).map{ Int(String($0))! }
    
    var i = digits.count - 1
    while i - 1 >= 0 && digits[i - 1] >= digits[i] {
        i -= 1
    }
    
    if i == 0 {
        return -1
    }
    
    var j = i
    while j + 1 < digits.count && digits[j + 1] > digits[i - 1] {
        j += 1
    }
    
    (digits[i - 1], digits[j]) = (digits[j], digits[i - 1])
    let leftPart = digits[0 ... i - 1]
    let rightPart = digits[i...]
    let assemble = Array(leftPart + rightPart.sorted())
    let ret = assemble.reduce(into: 0) {
        $0 *= 10
        $0 += $1
    }
    
    return ret < (1 << 31) ? ret : -1
}
```

[91. Decode Ways](https://leetcode.com/problems/decode-ways/)
``` swift
// DP
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func numDecodings(_ s: String) -> Int {
    let digits = Array(s).map { Int(String($0))! }
    var dp = [Int](repeating: 0, count: digits.count + 1)
    dp[0] = 1
    dp[1] = digits[0] != 0 ? 1 : 0
    if s.count > 1 {
        for i in 2 ... digits.count {
            if digits[i - 1] != 0 {
                dp[i] += dp[i - 1]
            }
            let k = digits[i - 2] * 10 + digits[i - 1]
            if k >= 10 && k <= 26 {
                dp[i] += dp[i - 2]
            }
        }
    }
    return dp[s.count]
}
```

[1463. Cherry Pickup II](https://leetcode.com/problems/cherry-pickup-ii/)
``` swift
// DP, top down
// 时间复杂度：O(m * n * n)
// 空间复杂度：O(m * n * n)
func cherryPickup(_ grid: [[Int]]) -> Int {
    let m = grid.count
    let n = grid[0].count
    var dp = [[[Int]]](repeating: [[Int]](repeating: [Int](repeating: -1, count: n), count: n), count: m)
    func dfs(_ r: Int, _ c1: Int, _ c2: Int) -> Int {
        guard r < m else { return 0 }
        guard dp[r][c1][c2] == -1 else { return dp[r][c1][c2] }
        var ret = 0
        for i in -1 ... 1 {
            for j in -1 ... 1 {
                let newC1 = c1 + i
                let newC2 = c2 + j
                if newC1 >= 0 && newC1 < n && newC2 >= 0 && newC2 < n {
                    ret = max(ret, dfs(r + 1, newC1, newC2))
                }
            }
        }
        let cherries = c1 == c2 ? grid[r][c1] : grid[r][c1] + grid[r][c2]
        dp[r][c1][c2] = ret + cherries
        return ret + cherries
    }
    return dfs(0, 0, n - 1)
}
```

[334. Increasing Triplet Subsequence](https://leetcode.com/problems/increasing-triplet-subsequence/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func increasingTriplet(_ nums: [Int]) -> Bool {
    guard nums.count >= 3 else { return false }
    var l1 = nums[0]
    var l2 = Int.max
    
    for i in 1 ..< nums.count {
        let n = nums[i]
        if n > l2 {
            return true
        } else if n < l1 {
            l1 = min(l1, n)
        } else if n > l1 {
            l2 = min(l2, n)
        }
    }
    
    return false
}
```

[865. Smallest Subtree with all the Deepest Nodes](https://leetcode.com/problems/smallest-subtree-with-all-the-deepest-nodes/)
``` swift
// DFS
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func subtreeWithAllDeepest(_ root: TreeNode?) -> TreeNode? {
    func dfs(_ root: TreeNode?) -> (Int, TreeNode?) {
        guard let node = root else { return (0, nil) }
        
        let left = dfs(node.left)
        let right = dfs(node.right)
        
        if left.0 == right.0 {
            return (left.0 + 1, root)
        } else {
            return (left.0 > right.0 ? left.0 + 1 : right.0 + 1,
                    left.0 > right.0 ? left.1 : right.1)
        }
    }
    
    return dfs(root).1
}
```

[117. Populating Next Right Pointers in Each Node II](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func connect(_ root: Node?) -> Node? {
    guard let root = root else { return nil }
    var cur: Node? = root
    var nextHead: Node?
    var nextNode: Node?
    while cur != nil {
        while cur != nil {
            if let left = cur?.left {
                if nextHead == nil {
                    nextHead = left
                    nextNode = left
                } else {
                    nextNode?.next = left
                    nextNode = nextNode?.next
                }
            }
            if let right = cur?.right {
                if nextHead == nil {
                    nextHead = right
                    nextNode = right
                } else {
                    nextNode?.next = right
                    nextNode = nextNode?.next
                }
            }
            cur = cur?.next
        }
        cur = nextHead
        nextHead = nil
        nextNode = nil
    }
    return root
}
```

[59. Spiral Matrix II](https://leetcode.com/problems/spiral-matrix-ii/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n*n)
func generateMatrix(_ n: Int) -> [[Int]] {
    var matrix = [[Int]](repeating: [Int](repeating: 0, count: n), count: n)
    var i = 0
    var j = 0
    var count = 1
    var dir = 0
    let range = 0 ..< n
    // 利用表做方向转换
    let direction = [
        [0, 1],
        [1, 0],
        [0, -1],
        [-1, 0]
    ]
    while count <= n * n {
        matrix[i][j] = count
        count += 1
        var row = i + direction[dir].first!
        var col = j + direction[dir].last!
        if !range.contains(row) || !range.contains(col) || matrix[row][col] != 0 {
            dir = (dir + 1) % 4
            row = i + direction[dir].first!
            col = j + direction[dir].last!
        }
        i = row
        j = col
    }
    return matrix
}
```

[382. Linked List Random Node](https://leetcode.com/problems/linked-list-random-node/)
``` swift
// Reservoir sampling
// 时间复杂度：O(n(1+log(N/n)))
// 空间复杂度：O(1)
class Solution {

    /** @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node. */
    let head: ListNode?

    init(_ head: ListNode?) {
        self.head = head
    }

    func getRandom() -> Int {
        var val = head!.val
        var node = head?.next
        var i = 2
        while node != nil {
            let target = Int.random(in: 0 ..< i)
            if target == 0 {
                val = node!.val
            }
            i += 1
            node = node?.next
        }
        return val
    }
}
```

[218. The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/)
``` swift
// 归并排序
// 时间复杂度：O(nlogn)
// 空间复杂度：O(n)
func getSkyline(_ buildings: [[Int]]) -> [[Int]] {
    func merge(_ A: [(Int, Int)], _ B: [(Int, Int)]) -> [(Int, Int)] {
        var ret = [(Int, Int)]()
        var h1 = 0
        var h2 = 0
        var i = 0
        var j = 0
        while i < A.count && j < B.count {
            var x = 0
            var h = 0

            if A[i].0 < B[j].0 {
                x = A[i].0
                h1 = A[i].1
                h = max(h1, h2)
                i += 1
            } else if A[i].0 > B[j].0 {
                x = B[j].0
                h2 = B[j].1
                h = max(h1, h2)
                j += 1
            } else {
                x = A[i].0
                h1 = A[i].1
                h2 = B[j].1
                h = max(h1, h2)
                i += 1
                j += 1
            }
            if ret.isEmpty || h != ret.last!.1 { // 高度有变化
                ret.append((x, h))
            }
        }
        while i < A.count {
            ret.append(A[i])
            i += 1
        }
        while j < B.count {
            ret.append(B[j])
            j += 1
        }
        return ret
    }

    func recurSkyline(_ A: [[Int]], _ s: Int, _ e: Int) -> [(Int, Int)] {
        if e > s {
            let mid = s + (e - s) / 2
            let B = recurSkyline(A, s, mid)
            let C = recurSkyline(A, mid + 1, e)
            return merge(B, C)
        } else {
            var v = [(Int, Int)]()
            v.append((A[s][0], A[s][2]))
            v.append((A[s][1], 0))
            return v
        }
    }

    guard buildings.count > 0 else { return buildings }

    return recurSkyline(buildings, 0, buildings.count - 1).map { [$0.0, $0.1] }
}
```

[1306. Jump Game III](https://leetcode.com/problems/jump-game-iii/)
``` swift
// BFS
// 时间复杂度：O(n)
// 空间复杂度：O(n)
// Runtime 100%
func canReach(_ arr: [Int], _ start: Int) -> Bool {
    var arr = arr
    func bfs(_ start: Int) -> Bool {
        var queue = [Int]()
        queue.append(start)
        while !queue.isEmpty {
            let cur = queue.removeFirst()
            if arr[cur] == 0 {
                return true
            }
            if arr[cur] < 0 {
                continue
            }
            if cur + arr[cur] < arr.count {
                queue.append(cur + arr[cur])
            }
            if cur - arr[cur] >= 0 {
                queue.append(cur - arr[cur])
            }
            arr[cur] = -1
        }
        return false
    }

    return bfs(start)
}
```

[239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)
``` swift
// 时间复杂度：O(K)
// 空间复杂度：O(n)
func maxSlidingWindow(_ nums: [Int], _ k: Int) -> [Int] {
    var maxLeft = [Int](repeating: 0, count: nums.count)
    var maxRight = [Int](repeating: 0, count: nums.count)
    maxLeft[0] = nums[0]
    maxRight[nums.count - 1] = nums[nums.count - 1]
    for i in 1 ..< nums.count {
        maxLeft[i] = (i % k == 0 ? nums[i] : max(maxLeft[i - 1], nums[i]))
        let j = nums.count - 1 - i
        maxRight[j] = j % k == 0 ? nums[j] : max(maxRight[j + 1], nums[j])
    }

    var ret = [Int](repeating: 0, count: nums.count - k + 1)
    for i in 0 ..< nums.count - k + 1 {
        ret[i] = max(maxRight[i], maxLeft[i + k - 1])
    }
    return ret

//        // 双向队列
//        guard k > 1 else { return nums }
//
//        var dequeue = [Int]()
//        var ret = [Int]()
//        for i in 0 ..< nums.count {
//            // 移除窗口外索引
//            while !dequeue.isEmpty && dequeue.first! < i - k + 1 {
//                dequeue.removeFirst()
//            }
//            // 保持元素单调递增
//            while !dequeue.isEmpty && nums[dequeue.last!] < nums[i] {
//                dequeue.removeLast()
//            }
//            dequeue.append(i)
//            if i >= k - 1 {
//                ret.append(nums[dequeue.first!])
//            }
//        }
//        return ret
}
```

[416. Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)
``` swift
// DP，bottom-up 比 top-down 更快
// 时间复杂度：O(K)
// 空间复杂度：O(n)
func canPartition(_ nums: [Int]) -> Bool {
    let sum = nums.reduce(0, +)
    guard sum % 2 == 0 else { return false }

    let target = sum / 2
    var dp = [Bool?](repeating: nil, count: target + 1) // optional 用于区分是否已被访问

    func bottomUp(_ i: Int, _ rest: Int) -> Bool {
        if rest == 0 {
            return true
        }

        if i >= nums.count || rest < 0 {
            return false
        }

        if let ret = dp[rest] {
            return ret
        }

        let ret = bottomUp(i + 1, rest - nums[i]) || bottomUp(i + 1, rest)
        dp[rest] = ret

        return ret
    }

    return bottomUp(0, target)
}
```

[1015. Smallest Integer Divisible by K](https://leetcode.com/problems/smallest-integer-divisible-by-k/)
``` swift
// 时间复杂度：O(K)
// 空间复杂度：O(n)
// Runtime 100%
func smallestRepunitDivByK(_ K: Int) -> Int {
    // 除 2 或 5 外，在 N <= K 时，都存在一个值（11...111，N 个 1）符合条件
    if K % 2 == 0 || K % 5 == 0 { return -1 } 
    var r = 0
    for N in 1 ... K {
        r = (r * 10 + 1) % K // a*b % m = ( a%m * b%m ) % m
        if r == 0 {
            return N
        }
    }
    return -1
}
```

[227. Basic Calculator II](https://leetcode.com/problems/basic-calculator-ii/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
// Runtime 100%
func calculate(_ s: String) -> Int {
    var stack = [Int]()
    var op = Character("+")
    var tmp = 0
    for c in s + "+" where !c.isWhitespace {
        if c.isNumber {
            tmp = tmp * 10 + c.wholeNumberValue!
        } else {
            if op == Character("+") {
                stack.append(tmp)
            } else if op == Character("-") {
                stack.append(-tmp)
            } else if op == Character("*") {
                stack.append(stack.removeLast() * tmp)
            } else if op == Character("/") {
                stack.append(stack.removeLast() / tmp)
            }
            op = c
            tmp = 0
        }
    }
    return stack.reduce(0) { $0 + $1 }
}
```

[337. House Robber III](https://leetcode.com/problems/house-robber-iii/)
``` swift
// DP
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func rob(_ root: TreeNode?) -> Int {
    func dfs(_ node: TreeNode?) -> (rob: Int, noRob: Int) {
        if node == nil {
            return (0, 0)
        }

        let left = dfs(node?.left)
        let right = dfs(node?.right)

        return (node!.val + left.1 + right.1, 
                max(left.0, left.1) + max(right.0, right.1))
    }
    let r = dfs(root)
    return max(r.0, r.1)
}
```

[902. Numbers At Most N Given Digit Set](https://leetcode.com/problems/numbers-at-most-n-given-digit-set/)
``` swift
// 时间复杂度：O(logn)
// 空间复杂度：O(1)
func atMostNGivenDigitSet(_ digits: [String], _ n: Int) -> Int {
    var ret = 0
    let digits = digits.map { Int($0)! }
    let n = String(n).map { Int(String($0))! }
    let nLen = n.count

    // 计算 x, xx, xxx（x 的个数少于 n 的位数时）的数量
    for i in 1 ..< nLen {
        ret += Int(pow(Double(digits.count), Double(i)))
    }
    // 计算 xxxx（x 的个数等于 n 的位数时）的数量
    for i in 0 ..< nLen {
        var hasSame = false // 如果 digits 里有 n 完全一样的数字时，最后结果需要加一
        for c in digits {
            if c < n[i] {
                ret += Int(pow(Double(digits.count), Double(nLen - i - 1)))
            } else if c == n[i] {
                hasSame = true
            }
        }
        if !hasSame {
            return ret
        }
    }

    return ret + 1
}
```

[81. Search in Rotated Sorted Array II](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/)
``` swift
// 时间复杂度：O(logn)
// 空间复杂度：O(1)
func search(_ nums: [Int], _ target: Int) -> Bool {
    var l = 0
    var h = nums.count - 1
    while l <= h {
        let mid = l + (h - l) / 2
        if nums[mid] == target {
            return true
        }

        if nums[mid] == nums[l] {
            l += 1
        } else if nums[mid] == nums[h] {
            h -= 1
        } else if nums[mid] > nums[h] {
            if target > nums[mid] || target <= nums[h] {
                l = mid + 1
            } else {
                h = mid - 1
            }
        } else {
            if  nums[mid] < target && target <= nums[h] {
                l = mid + 1
            } else {
                h = mid - 1
            }
        }
    }
    return false
}
```

[56. Merge Intervals](https://leetcode.com/problems/merge-intervals/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func merge(_ intervals: [[Int]]) -> [[Int]] {
    guard intervals.count > 1 else { return intervals }
    let intervals = intervals.sorted { $0[0] < $1[0] }
    var ret = [[Int]]()
    var cur = intervals[0]
    for i in 1 ..< intervals.count {
        let intr = intervals[i]
        if intr[0] <= cur[1] {
            cur[1] = max(cur[1], intr[1])
        } else {
            ret.append(cur)
            cur = intr
        }
    }
    ret.append(cur)
    return ret
}
```

[845. Longest Mountain in Array](https://leetcode.com/problems/longest-mountain-in-array/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func longestMountain(_ A: [Int]) -> Int {
    guard A.count >= 3 else { return 0 }

    var ret = 0
    var flag = -1
    var start = -1
    for i in 1 ..< A.count {
        if flag == -1 {
            if A[i] > A[i - 1] {
                start = i - 1
                flag = 0
            }
        } else if flag == 0 {
            if A[i] > A[i - 1] {
                // increasing
            } else if A[i] == A[i - 1] {
                start = -1
                flag = -1
            } else {
                // start decrease
                flag = 1
            }
        } else if flag == 1 {
            if A[i] < A[i - 1] {
                // decreasing
            } else {
                if start >= 0 {
                    ret = max(ret, i - start)
                }
                if A[i] > A[i - 1] {
                    start = i - 1
                    flag = 0
                } else {
                    start = -1
                    flag = -1
                }
            }
        }
    }
    if flag == 1 && start >= 0 {
        ret = max(ret, A.count - start)
    }
    return ret
}
```

[458. Poor Pigs](https://leetcode.com/problems/poor-pigs/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func poorPigs(_ buckets: Int, _ minutesToDie: Int, _ minutesToTest: Int) -> Int {
    var pig = 0
    while pow(Decimal((minutesToTest / minutesToDie + 1)), pig) < Decimal(buckets) {
        pig += 1
    }
    return pig
}
```

[116. Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func connect(_ root: Node?) -> Node? {
    guard let root = root else { return nil }
    var pre: Node? = nil
    var cur: Node? = root
    while cur?.left != nil {
        pre = cur
        while pre != nil {
            if pre?.left != nil {
                pre?.left?.next = pre?.right
            }
            if pre?.next != nil {
                pre?.right?.next = pre?.next?.left
            }
            pre = pre?.next // 通过 next 在层里遍历
        }
        cur = cur?.left // 到达每层最左节点
    }
    return root
}
```

[47. Permutations II](https://leetcode.com/problems/permutations-ii/)
``` swift
// 利用二分搜索加快速度
// 时间复杂度：O(n!)
// 空间复杂度：O(n)
func permuteUnique(_ nums: [Int]) -> [[Int]] {
    var ret = [[Int]]()
    let nums = nums.sorted()

    func permute(_ i: Int, _ nums: [Int]) {
        if i == nums.count - 1 {
            ret.append(nums)
            return
        }

        var n = nums // 保存当前交换结果
        for k in i ..< nums.count {
            if i != k && n[i] == n[k] {
                continue
            }
            n.swapAt(i, k)
            permute(i + 1, n)
        }
    }
    permute(0, nums)
    return ret
}
```

[1283. Find the Smallest Divisor Given a Threshold](https://leetcode.com/problems/find-the-smallest-divisor-given-a-threshold/)
``` swift
// 利用二分搜索加快速度
// 时间复杂度：O(nlogn)
// 空间复杂度：O(1)
func smallestDivisor(_ nums: [Int], _ threshold: Int) -> Int {
    var l = 1
    var r: Int = {
        var max = 0
        for n in nums where n > max {
            max = n
        }
        return max
    }()
    while l <= r {
        let mid = l + (r - l) / 2
        var sum = 0
        for n in nums {
            sum += n / mid + (n % mid > 0 ? 1 : 0)
            if sum > threshold {
                break
            }
        }
        if sum <= threshold {
            r = mid - 1
        } else {
            l = mid + 1
        }
    }

    return l
}
```

[310. Minimum Height Trees](https://leetcode.com/problems/minimum-height-trees/)
``` swift
// 利用拓扑排序，一层层排除叶子节点，直到最后只剩下不超过 2 个节点
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func findMinHeightTrees(_ n: Int, _ edges: [[Int]]) -> [Int] {
    if n < 2 {
        var centroids = [Int]()
        for i in 0 ..< n {
            centroids.append(i)
        }
        return centroids
    }

    var neighbors = [Set<Int>](repeating: Set<Int>(), count: n)
    for e in edges {
        neighbors[e[0]].insert(e[1])
        neighbors[e[1]].insert(e[0])
    }

    var leaves = [Int]()
    for i in 0 ..< n where neighbors[i].count == 1 {
        leaves.append(i)
    }
    var remainingNodes = n
    while remainingNodes > 2 {
        remainingNodes -= leaves.count
        var newLeaves = [Int]()
        for l in leaves {
            for n in neighbors[l] {
                neighbors[n].remove(l)
                if neighbors[n].count == 1 {
                    newLeaves.append(n)
                }
            }
        }
        leaves = newLeaves
    }
    return leaves
}
```

[147. Insertion Sort List](https://leetcode.com/problems/insertion-sort-list/)
``` swift
// 时间复杂度：O(n * n)
// 空间复杂度：O(1)
func insertionSortList(_ head: ListNode?) -> ListNode? {
    guard head != nil && head?.next != nil else { return head }

    let newHead = ListNode(Int.min)
    newHead.next = head

    var pre = head
    var cur = head?.next
    while cur != nil {
        if cur!.val < pre!.val {
            pre!.next = cur?.next

            var node = newHead
            while let next = node.next, cur!.val > next.val {
                node = next
            }
            let tmp = node.next
            node.next = cur
            cur?.next = tmp

            cur = pre!.next
        } else {
            pre = cur!
            cur = pre!.next
        }
    }

    return newHead.next
}
```

[1314. Matrix Block Sum](https://leetcode.com/problems/matrix-block-sum/)
``` swift
// DP
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func matrixBlockSum(_ mat: [[Int]], _ K: Int) -> [[Int]] {
    let m = mat.count
    let n = mat[0].count
    var anw = [[Int]](repeating: [Int](repeating: 0, count: n), count: m)
    var accu = [[Int]](repeating: [Int](repeating: 0, count: n + 1), count: m + 1) // Running Total
    for i in 1 ... m {
        for j in 1 ... n {
            accu[i][j] = mat[i - 1][j - 1] + accu[i - 1][j] + accu[i][j - 1] - accu[i - 1][j - 1]
        }
    }

    for i in 0 ..< m {
        for j in 0 ..< n {
            let rowMin = max(i - K, 0)
            let colMin = max(j - K, 0)
            let rowMax = min(i + K + 1, m)
            let colMax = min(j + K + 1, n)
            anw[i][j] = accu[rowMax][colMax] - accu[rowMin][colMax] - accu[rowMax][colMin] + accu[rowMin][colMin]
        }
    }

    return anw
}
```

[673. Number of Longest Increasing Subsequence](https://leetcode.com/problems/number-of-longest-increasing-subsequence/)
``` swift
// DP
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func findNumberOfLIS(_ nums: [Int]) -> Int {
    guard nums.count > 0 else { return 0 }

    var length = [Int](repeating: 1, count: nums.count) // length[i] 表示到 i 为止的最长单调上升子序列的长度
    var count = [Int](repeating: 1, count: nums.count) // count[i] 表示到 i 为止的最长单调上升子序列的个数
    var maxLength = 0
    var ret = 0
    for i in 0 ..< nums.count {
        for j in 0 ..< i where nums[j] < nums[i] {
            if length[j] + 1 > length[i] {
                length[i] = length[j] + 1
                count[i] = count[j]
            } else if length[j] + 1 == length[i] {
                count[i] += count[j]
            }
        }
        if length[i] == maxLength {
            ret += count[i]
        } else if length[i] > maxLength {
            ret = count[i]
            maxLength = length[i]
        }
    }

    return ret
}
```

[849. Maximize Distance to Closest Person](https://leetcode.com/problems/maximize-distance-to-closest-person/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func maxDistToClosest(_ seats: [Int]) -> Int {
    var prev = -1
    var maxDist = 0
    for i in 0 ..< seats.count where seats[i] == 1 {
        maxDist = max(maxDist, prev == -1 ? i : (i - prev) / 2)
        prev = i
    }
    return max(seats.count - prev - 1, maxDist)
}
```

[142. Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)
``` swift
// 双指针
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func detectCycle(_ head: ListNode?) -> ListNode? {
    // 1、判断是否有环
    var slow = head
    var fast = head?.next
    while fast != nil {
        if slow! === fast! {
            // 2、跑一圈，获取环长度
            var ptr = slow?.next
            var idx = 1
            while slow! !== ptr! {
                idx += 1
                ptr = ptr?.next
            }

            // 3.1、先让 second 跑一个环长度
            var first = head
            var second = head
            while idx > 0 {
                second = second?.next
                idx -= 1
            }

            // 3.2、两个一起跑，直到相遇
            while first! !== second! {
                first = first?.next
                second = second?.next
            }

            return first
        }
        slow = slow?.next
        fast = fast?.next?.next
    }

    return nil
}
```

[799. Champagne Tower](https://leetcode.com/problems/champagne-tower/)
``` swift
// DP
// 时间复杂度：O(R * R)
// 空间复杂度：O(R * R)
func champagneTower(_ poured: Int, _ query_row: Int, _ query_glass: Int) -> Double {
    var dp = [[Double]](repeating: [Double](repeating: 0, count: query_row + 2), count: query_row + 2)
    dp[0][0] = Double(poured)

    for i in 0 ... query_row {
        for j in 0 ... i where dp[i][j] > 1 {
            dp[i + 1][j] += (dp[i][j] - 1) / 2
            dp[i + 1][j + 1] += (dp[i][j] - 1) / 2
            dp[i][j] = 1
        }
    }

    return dp[query_row][query_glass]
}
```

[1510. Stone Game IV](https://leetcode.com/problems/stone-game-iv/)
``` swift
// DP
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func winnerSquareGame(_ n: Int) -> Bool {
    var dp = [Bool](repeating: false, count: n + 1)
    for i in 1 ... n {
        // 只要找到 i - k * k 前为 false，那么当前第 i 步即可为赢
        for k in 1 ... Int(sqrt(Double(n))) where i - k * k >= 0 {
            if !dp[i - k * k] {
                dp[i] = true
                break
            }
        }
    }
    return dp[n]
}
```

[948. Bag of Tokens](https://leetcode.com/problems/bag-of-tokens/)
``` swift
// 贪心
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func bagOfTokensScore(_ tokens: [Int], _ P: Int) -> Int {
    let tokens = tokens.sorted()
    var score = 0
    var maxScore = 0
    var power = P
    var i = 0
    var j = tokens.count - 1
    while i <= j {
        if power >= tokens[i] {
            power -= tokens[i]
            i += 1
            score += 1
            maxScore = max(maxScore, score)
        } else if score > 0 {
            let p = tokens[j]
            j -= 1
            power += p
            score -= 1
        } else {
            break
        }
    }

    return maxScore
}
```

[456. 132 Pattern](https://leetcode.com/problems/132-pattern/)
``` swift
// Stack
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func find132pattern(_ nums: [Int]) -> Bool {
    guard nums.count > 2 else { return false }

    // 记录 nums[i] 前的最小值
    var mins = [Int](repeating: nums[0], count: nums.count)
    for i in 1 ..< nums.count {
        mins[i] = min(mins[i - 1], nums[i])
    }

    var stack = [Int]()
    for j in stride(from: nums.count - 1, to: -1, by: -1) where nums[j] > mins[j] {
        while !stack.isEmpty && stack.last! <= mins[j] {
            stack.removeLast()
        }
        if !stack.isEmpty && stack.last! < nums[j] && stack.last! > mins[j] {
            return true
        }
        stack.append(nums[j])
    }

    return false
}
```

[133. Clone Graph](https://leetcode.com/problems/clone-graph/)
``` swift
// DFS
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func cloneGraph(_ node: Node?) -> Node? {
    var dict = [Int: Node]() // 用于处理出现环时

    func clone(_ node: Node?) -> Node? {
        guard let node = node else { return nil }

        if let visited = dict[node.val] {
            return visited
        }

        let copy = Node(node.val)
        var neighbors = [Node?]()
        dict[copy.val] = copy
        for n in node.neighbors {
            if let clone = cloneGraph(n) {
                neighbors.append(clone)
            }
        }
        copy.neighbors = neighbors

        return copy
    }

    return clone(node)
}
```

[188. Best Time to Buy and Sell Stock IV](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/)
``` swift
// DP
// 时间复杂度：O(n * k)
// 空间复杂度：O(n * k)
func maxProfit(_ k: Int, _ prices: [Int]) -> Int {
    guard prices.count > 2 && k > 0 else { return 0 }

    if k >= prices.count / 2 {
        var ret = 0
        for i in 1 ..< prices.count where prices[i] > prices[i - 1] {
            ret += prices[i] - prices[i - 1]
        }
        return ret
    }

    var dp = [[Int]](repeating: [Int](repeating: 0, count: prices.count), count: k + 1)
    var localMax = Int.min
    for i in 1 ... k {
        for j in 1 ..< prices.count {
            localMax = max(localMax, dp[i - 1][j - 1] - prices[j - 1])
            // localMax + prices[j]: max(dp[i - 1][jj - 1] + prices[j] - price[jj]), jj = 1 ..< j
            dp[i][j] = max(dp[i][j - 1], localMax + prices[j])
        }
        localMax = Int.min
    }
    return dp[k][prices.count - 1]
}
```

[187. Repeated DNA Sequences](https://leetcode.com/problems/repeated-dna-sequences/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func findRepeatedDnaSequences(_ s: String) -> [String] {
    guard s.count > 10 else { return [] }
    let dict = [Character("A"): 0,
                Character("C"): 1,
                Character("G"): 2,
                Character("T"): 3]
    let chars = Array(s)
    let values = chars.map { dict[$0]! } // 映射，用于后面的位运算
    let mask = (1 << 20) - 1
    var set1 = Set<Int>()
    var set2 = Set<Int>() // 用双集合判断是否已存在并只存在一次
    var ret = [String]()
    var val = 0
    for j in 0 ..< 10 {
        val <<= 2
        val |= Int(values[j])
    }
    set1.insert(val)
    for i in 10 ..< values.count {
        val = (val << 2 & mask) | values[i] // 加速，每次只处理一个值，不再用10个值重新生成

        if !set1.insert(val).inserted && set2.insert(val).inserted {
            ret.append(String(chars[i - 9 ... i]))
        }
    }
    return ret
}
```

[74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)
``` swift
// 二分搜索
// 时间复杂度：O(log(m) + log(n))
// 空间复杂度：O(1)
func searchMatrix(_ matrix: [[Int]], _ target: Int) -> Bool {
    guard matrix.count > 0 && matrix[0].count > 0 else { return false }

    var i = 0
    var j = matrix.count - 1
    while i <= j {
        let m = i + (j - i) / 2
        if matrix[m][0] == target {
            return true
        } else if matrix[m][0] > target {
            j = m - 1
        } else {
            i = m + 1
        }
    }

    i = i > 0 ? i - 1 : 0
    var l = 0
    var r = matrix[0].count - 1
    while l <= r {
        let m = l + (r - l) / 2
        if matrix[i][m] == target {
            return true
        } else if matrix[i][m] > target {
            r = m - 1
        } else {
            l = m + 1
        }
    }

    return false
}
```

[213. House Robber II](https://leetcode.com/problems/house-robber-ii/)
``` swift
// DP
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func rob(_ nums: [Int]) -> Int {
    func findMax(_ nums: ArraySlice<Int>) -> Int {
        var prev1 = 0
        var prev2 = 0
        for n in nums {
            let tmp = prev1
            prev1 = max(prev2 + n, prev1)
            prev2 = tmp
        }
        return prev1
    }

    return max(findMax(nums.dropFirst()), findMax(nums.dropLast()), nums.first ?? 0)
}
```

[316. Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/)
``` swift
// DP
// 时间复杂度：O(n^2)
// 空间复杂度：O(n)
func removeDuplicateLetters(_ s: String) -> String {
    let s = Array(s)
    var lastIdx = [Int](repeating: 0, count: 26)
    let base = Int(Character("a").asciiValue!)
    for (idx, c) in s.enumerated() {
        lastIdx[Int(c.asciiValue!) - base] = idx
    }
    var seen = [Bool](repeating: false, count: 26)
    var stack = [Int]()
    for (idx, c) in s.enumerated() {
        let val = Int(c.asciiValue!) - base
        if seen[val] {
            continue
        }

        while !stack.isEmpty && stack.last! > val && idx < lastIdx[stack.last!] {
            seen[stack.removeLast()] = false
        }
        
        stack.append(val)
        seen[val] = true
    }

    return String(stack.map { Character(UnicodeScalar(UInt8($0 + base))) })
}
```

[61. Rotate List](https://leetcode.com/problems/rotate-list/)
``` swift
// 双指针
// 时间复杂度：O(n)
func rotateRight(_ head: ListNode?, _ k: Int) -> ListNode? {
    guard head != nil && k > 0 else { return head }

    var fast = head
    var slow = head
    var count = 0
    while true {
        if count == k {
            break
        } else if fast?.next != nil {
            fast = fast?.next
            count += 1
        } else {
            return rotateRight(head, k % (count + 1))
        }
    }

    while fast?.next != nil {
        fast = fast?.next
        slow = slow?.next
    }

    let newHead = slow?.next
    slow?.next = nil
    fast?.next = head
    return newHead
}
```

[139. Word Break](https://leetcode.com/problems/word-break/)
``` swift
// DP
// 时间复杂度：O(m * n)
// 空间复杂度：O(n)
func wordBreak(_ s: String, _ wordDict: [String]) -> Bool {
    let wordDict = Set<String>(wordDict)
    var dp = [Bool](repeating: false, count: s.count + 1)
    dp[0] = true

    // forward
//        for i in 1 ... s.count {
//            for j in 0 ..< i {
//                let str = s[s.index(s.startIndex, offsetBy: j) ..< s.index(s.startIndex, offsetBy: i)]
//                if dp[j] && wordDict.contains(String(str)) {
//                    dp[i] = true
//                }
//            }
//        }

    // backward
    for i in 1 ... s.count {
        let suffix = s.suffix(i)
        for w in wordDict {
            if i >= w.count, dp[i - w.count], suffix.hasPrefix(w) {
                dp[i] = true
            }
        }
    }

    return dp.last!
}
```

[980. Unique Paths III](https://leetcode.com/problems/unique-paths-iii/)
``` swift
// 时间复杂度：O(m * n)
// 空间复杂度：O(m * n)
func uniquePathsIII(_ grid: [[Int]]) -> Int {
    var grid = grid
    var res = 0
    var start = (0, 0)
    var count = 0
    for i in 0 ..< grid.count {
        for j in 0 ..< grid[0].count {
            if grid[i][j] == 0 {
                count += 1
            } else if grid[i][j] == 1 {
                start = (i, j)
            }
        }
    }

    func dfs(_ p: Int, _ q: Int, _ step: Int) {
        guard p >= 0 && p < grid.count && q >= 0 && q < grid[0].count else { return }
        if step == count + 1 && grid[p][q] == 2 {
            res += 1
        } else if grid[p][q] == 0 || grid[p][q] == 1 {
            grid[p][q] = -1
            dfs(p + 1, q, step + 1)
            dfs(p - 1, q, step + 1)
            dfs(p, q + 1, step + 1)
            dfs(p, q - 1, step + 1)
            grid[p][q] = 0
        }
    }

    dfs(start.0, start.1, 0)

    return res
}
```

[1291. Sequential Digits](https://leetcode.com/problems/sequential-digits/)
``` swift
// 时间复杂度：O(len(hight))
// 空间复杂度：O(n)
func sequentialDigits(_ low: Int, _ high: Int) -> [Int] {
    let highLength = String(high).count
    var res = [Int]()
    for i in 1 ... 9 {
        var n = i
        var c = 1
        while c < highLength && i + c < 10 {
            n = n * 10 + i + c
            c += 1

            if n >= low && n <= high {
                res.append(n)
            }
        }
    }
    return res.sorted()
}
```

[949. Largest Time for Given Digits](https://leetcode.com/problems/largest-time-for-given-digits/)
``` swift
// 时间复杂度：O(1)
// 空间复杂度：O(1)
func largestTimeFromDigits(_ A: [Int]) -> String {
    var minute = -1
    var hour = -1
    var digits = A

    func permute(_ i: Int) {
        if i == 4 {
            let h = digits[0] * 10 + digits[1]
            let m = digits[2] * 10 + digits[3]
            if h <= 23 && m <= 59 && (h > hour || (h == hour && m > minute)) {
                hour = h
                minute = m
            }
            return
        }

        for k in i ..< 4 {
            (digits[i], digits[k]) = (digits[k], digits[i])
            permute(i + 1)
            (digits[i], digits[k]) = (digits[k], digits[i])
        }
    }

    // 遍历所有组合
    permute(0)

    if hour > -1 && minute > -1 {
        return "\(hour / 10)\(hour % 10):\(minute / 10)\(minute % 10)"
    }

    return ""
}
```

[497. Random Point in Non-overlapping Rectangles](https://leetcode.com/problems/random-point-in-non-overlapping-rectangles/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
class Solution {
    var map = [Int]()
    var rects = [[Int]]()

    init(_ rects: [[Int]]) {
        self.rects = rects
        var sum = 0
        for r in rects {
            // 这里算的不是面积,而是可以选择的点的数量
            let points = (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
            sum += points
            map.append(sum)
        }
    }

    func pick() -> [Int] {
        // 在 [1 ... 总点数] 的范围内先随机选择对应的矩形
        let r = Int.random(in: 1 ... map.last!)
        var rect = [Int]()
        for (i, p) in map.enumerated() where p >= r {
            rect = rects[i]
            break
        }
        // 最后返回矩形内的点位置
        return [
            Int.random(in: rect[0] ... rect[2]),
            Int.random(in: rect[1] ... rect[3])
        ]
    }
}
```

[1286. Iterator for Combination](https://leetcode.com/problems/iterator-for-combination/)
``` swift
// 时间复杂度：O(n * m)
// 空间复杂度：O(n)
class CombinationIterator {

    var idx = 0
    var set = [String]()

    init(_ characters: String, _ combinationLength: Int) {
        let chars = Array(characters)
        func iterate(_ c: String, _ i: Int) {
            if c.count == combinationLength {
                set.append(String(c))
                return
            }
            for j in i ..< characters.count {
                iterate(c + "\(chars[j])", j + 1)
            }
        }

        iterate("", 0)
    }

    func next() -> String {
        let ret = set[idx]
        idx += 1
        return ret
    }

    func hasNext() -> Bool {
        return idx < set.count
    }
    
}
```

[987. Vertical Order Traversal of a Binary Tree](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func verticalTraversal(_ root: TreeNode?) -> [[Int]] {
    guard let root = root else { return [] }

    var dict = [Int: [(Int, Int)]]() // [x: [(val, y)]]
    var stack = [(root, 0, 0)] // [(node, x, y)]
    while !stack.isEmpty {
        let count = stack.count
        for _ in 0 ..< count {
            let node = stack.removeFirst()

            dict[node.1, default: []].append((node.0.val, node.2))

            if let left = node.0.left {
                stack.append((left, node.1 - 1, node.2 - 1))
            }
            if let right = node.0.right {
                stack.append((right, node.1 + 1, node.2 - 1))
            }
        }
    }

    var res = [[Int]]()
    for k in dict.keys.sorted() {
        res.append(dict[k]!.sorted(by: { (a, b) -> Bool in
            if a.1 != b.1 {
                return a.1 >= b.1
            } else {
                return a.0 <= b.0
            }
        }).map{ $0.0 })
    }
    return res
}
```

[309. Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)
``` swift
// DP,对可能出现的状态转移都进行处理
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func maxProfit(_ prices: [Int]) -> Int {
    var buy = Int.min
    var sell = 0
    var prevBuy = 0
    var prevSell = 0
    for p in prices {
        prevBuy = buy
        buy = max(prevSell - p, prevBuy)
        prevSell = sell
        sell = max(prevBuy + p, sell)
    }

    return sell
}
```

[106. Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func buildTree(_ inorder: [Int], _ postorder: [Int]) -> TreeNode? {
    var index = [Int: Int]()
    for i in 0 ..< inorder.count {
        index[inorder[i]] = i
    }

    var cur = postorder.count - 1

    func build(_ left: Int, _ right: Int) -> TreeNode? {
        guard left <= right else { return nil }
        let node = TreeNode(postorder[cur])
        let idx = index[node.val]!
        cur -= 1
        node.right = build(idx + 1, right)
        node.left = build(left, idx - 1)
        return node
    }

    return build(0, cur)
}
```

[15. 3Sum](https://leetcode.com/problems/3sum/)
``` swift
// 时间复杂度：O(n * n)
// 空间复杂度：O(n)
func threeSum(_ nums: [Int]) -> [[Int]] {
    guard nums.count > 2 else { return [] }

    let nums = nums.sorted()
    var res = [[Int]]()
    for i in 0 ..< nums.count - 2 {
        if nums[i] > 0 {
            break
        }
        // 在剩余的数组里双向搜索目标
        if i == 0 || (i > 0 && nums[i] != nums[i - 1]) {
            var l = i + 1
            var h = nums.count - 1
            while l < h {
                if nums[l] + nums[h] == -nums[i] {
                    res.append([nums[i], nums[l], nums[h]])

                    // 去重
                    while l < h && nums[l] == nums[l + 1] {
                        l += 1
                    }
                    while l < h && nums[h] == nums[h - 1] {
                        h -= 1
                    }
                    l += 1
                    h -= 1
                } else if nums[l] + nums[h] > -nums[i] {
                    h -= 1
                } else {
                    l += 1
                }
            }
        }
    }

    return res
}
```

[264. Ugly Number II](https://leetcode.com/problems/ugly-number-ii/)
``` swift
// DP
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func nthUglyNumber(_ n: Int) -> Int {
    var res = [Int](repeating: 1, count: n)
    var k1 = 0, k2 = 0, k3 = 0
    for i in 1 ..< n {
        let v1 = res[k1] * 2
        let v2 = res[k2] * 3
        let v3 = res[k3] * 5
        let next = min(v1, min(v2, v3))

        if next == v1 {
            k1 += 1
        }
        if next == v2 {
            k2 += 1
        }
        if next == v3 {
            k3 += 1
        }
        res[i] = next
    }
    return res[n - 1]
}
```

[332. Reconstruct Itinerary](https://leetcode.com/problems/reconstruct-itinerary/)
``` swift
// Stack 处理 DFS
// 时间复杂度：O(m * n)
// 空间复杂度：O(m)
func findItinerary(_ tickets: [[String]]) -> [String] {
    var dict = tickets.reduce(into: [String: [String]]()) {
        var array = $0[$1.first!] ?? [String]()
        array.append($1.last!)
        $0[$1.first!] = array
    }
    
    for k in dict.keys {
        dict[k]?.sort()
    }

    var res = [String]()
    var stack = ["JFK"]
    while stack.count > 0 {
        let key = stack.last!
        if dict[key]?.count ?? 0 > 0 {
            stack.append(dict[key]!.removeFirst())
        } else {
            // 这个机场已经不能去任何地方，所以加到最终行程里
            res.append(stack.removeLast())
        }

    }
    return res.reversed()
}
```

[62. Unique Paths](https://leetcode.com/problems/unique-paths/)
``` swift
// 时间复杂度：O(m * n)
// 空间复杂度：O(m)
func uniquePaths(_ m: Int, _ n: Int) -> Int {
    var memo = [Int](repeating: 0, count: m)        
    memo[0] = 1
    for r in 0 ..< n {
        for c in 1 ..< m {
            memo[c] = memo[c] + memo[c - 1]
        }
    }
    return memo[m - 1]
}
```

[279. Perfect Squares](https://leetcode.com/problems/perfect-squares/)
``` swift
// DP
// 时间复杂度：O(n * n)
// 空间复杂度：O(n)
func numSquares(_ n: Int) -> Int {
    var dp = [Int](repeating: 0, count: n + 1)
    for j in 1 ... n {
        var k = Int.max
        var i = 1
        while i * i <= j {
            k = min(k, dp[j - i * i] + 1)
            i += 1
        }
        dp[j] = k
    }
    return dp[n]
}
```

[174. Dungeon Game](https://leetcode.com/problems/dungeon-game/)
``` swift
// DP
// 时间复杂度：O(n * m)
// 空间复杂度：O(n * m)
func calculateMinimumHP(_ dungeon: [[Int]]) -> Int {
    guard dungeon.count > 0 else { return 0 }

    let row = dungeon.count
    let col = dungeon[0].count

    var dp = [[Int]](repeating: [Int](repeating: Int.max / 2, count: col + 1), count: row + 1)
    dp[row - 1][col] = 1
    dp[row][col - 1] = 1

    for i in stride(from: row - 1, to: -1, by: -1) {
        for j in stride(from: col - 1, to: -1, by: -1) {
            let right = max(dp[i][j + 1] - dungeon[i][j], 1)
            let down = max(dp[i + 1][j] - dungeon[i][j], 1)
            dp[i][j] = min(right, down)
        }
    }
    return dp[0][0]
}
```

[60. Permutation Sequence](https://leetcode.com/problems/permutation-sequence/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func getPermutation(_ n: Int, _ k: Int) -> String {
    var nums = Array(1 ... n)
    var res = ""
    var k = k

    while nums.count > 1 {
        let times = Array(1 ..< nums.count).reduce(1, *)
        let i = (k - 1) / times
        k -= i * times
        res += "\(nums.remove(at: i))"
    }

    for t in nums {
        res += "\(t)"
    }

    return res
}
```

[130. Surrounded Regions](https://leetcode.com/problems/surrounded-regions/)
``` swift
// DFS，逆向思维，把四边不符合要求的格子先标出来，最后统一处理
// 时间复杂度：O(m * n)
// 空间复杂度：O(m * n)
func solve(_ board: inout [[Character]]) {
    guard board.count > 0 else { return }

    func dfs(_ board: inout [[Character]], _ i: Int, _ j: Int) {
        if i < 0 || j < 0 || i >= board.count || j >= board[0].count || board[i][j] != Character("O") {
            return
        }
        board[i][j] = Character("1")
        dfs(&board, i + 1, j)
        dfs(&board, i - 1, j)
        dfs(&board, i, j + 1)
        dfs(&board, i, j - 1)
    }

    let row = board.count
    let column = board[0].count

    // 上下两边
    for j in 0 ..< column {
        if board[0][j] == Character("O") {
            dfs(&board, 0, j)
        }
        if board[row - 1][j] == Character("O") {
            dfs(&board, row - 1, j)
        }
    }
    // 左右两边
    for i in 0 ..< row {
        if board[i][0] == Character("O") {
            dfs(&board, i, 0)
        }
        if board[i][column - 1] == Character("O") {
            dfs(&board, i, column - 1)
        }
    }

    for i in 0 ..< row {
        for j in 0 ..< column {
            if board[i][j] == Character("O") {
                board[i][j] = Character("X")
            } else if board[i][j] == Character("1") {
                board[i][j] = Character("O")
            }
        }
    }
}
```

[787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/)
``` swift
// DP
// 时间复杂度：O(m *n)
// 空间复杂度：O(m * n)
func findCheapestPrice(_ n: Int, _ flights: [[Int]], _ src: Int, _ dst: Int, _ K: Int) -> Int {
    var dp = [[Int]](repeating: [Int](repeating: Int.max, count: n), count: K + 2)
    for i in 0 ... K + 1 {
        dp[i][src] = 0
    }
    for i in 1 ... K + 1 {
        for f in flights {
            let s = f[0]
            let e = f[1]
            let w = f[2]
            if dp[i - 1][s] != Int.max {
                dp[i][e] = min(dp[i][e], dp[i - 1][s] + w)
            }
        }
    }
    return dp[K + 1][dst] == Int.max ? -1 : dp[K + 1][dst]
}
```

[368. Largest Divisible Subset](https://leetcode.com/problems/largest-divisible-subset/)
``` swift
// DP
// 时间复杂度：O(n ^ 2)
// 空间复杂度：O(n)
func largestDivisibleSubset(_ nums: [Int]) -> [Int] {
    let nums = nums.sorted()
    var count = [Int](repeating: 0, count: nums.count)
    var pre = [Int](repeating: 0, count: nums.count)
    var max = 0
    var idx = -1
    for i in 0 ..< nums.count {
        count[i] = 1
        pre[i] = -1
        for j in stride(from: i - 1, to: -1, by: -1) {
            if nums[i] % nums[j] == 0 && 1 + count[j] > count[i] {
                count[i] = count[j] + 1
                pre[i] = j
            }
        }
        if count[i] > max {
            max = count[i]
            idx = i
        }
    }

    var res = [Int]()
    while idx != -1 {
        res.append(nums[idx])
        idx = pre[idx]
    }
    return res
}
```

[380. Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/)
``` swift
class RandomizedSet {

    var queue = [Int]()
    var dict = [Int: Int]()

    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    func insert(_ val: Int) -> Bool {
        guard dict[val] == nil else {
            return false
        }

        queue.append(val)
        dict[val] = queue.count - 1

        return true
    }

    /** Removes a value from the set. Returns true if the set contained the specified element. */
    func remove(_ val: Int) -> Bool {
        guard let idx = dict[val] else {
            return false
        }

        // 把最后一个值跟目标值替换
        let last = queue.last!
        queue[idx] = last
        dict[last] = idx
        dict[val] = nil
        queue.removeLast()

        return true
    }

    /** Get a random element from the set. */
    func getRandom() -> Int {
        guard !queue.isEmpty else {
            return -1
        }
        return queue[Int.random(in: 0 ..< queue.count)]
    }
}
```

[518. Coin Change 2](https://leetcode.com/problems/coin-change-2/)
``` swift
// 时间复杂度：O(m * n)
// 空间复杂度：O(n)
func change(_ amount: Int, _ coins: [Int]) -> Int {
    var dp = [Int](repeating: 0, count: amount + 1)
    dp[0] = 1
    for i in coins {
        for j in 1 ..< amount + 1 {
            if j >= i {
                dp[j] += dp[j - i]
            }
        }
    }

    return dp[amount]
}
```

[226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func invertTree(_ root: TreeNode?) -> TreeNode? {
    var stack = [root]
    while !stack.isEmpty {
        if let cur = stack.removeLast() {
            (cur.left, cur.right) = (cur.right, cur.left)
            stack.append(cur.left)
            stack.append(cur.right)
        }
    }
    return root
}

func invertTree(_ root: TreeNode?) -> TreeNode? {
    guard let root = root else { return nil }
    (root.left, root.right) = (root.right, root.left)
    invertTree(root.left)
    invertTree(root.right)
    return root
}
```
[207. Course Schedule](https://leetcode.com/problems/course-schedule/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(n)
func canFinish(_ numCourses: Int, _ prerequisites: [[Int]]) -> Bool {
    var numCourses = numCourses
    var adj = [[Int]](repeating: [Int](), count: numCourses)
    var degree = [Int](repeating: 0, count: numCourses)
    // 初始化邻接表和入度数组
    for p in prerequisites {
        adj[p[1]].append(p[0])
        degree[p[0]] += 1
    }
    // BFS 求拓扑排序
    // 把所有入度为0顶点加入队列
    var queue = [Int]()
    for i in 0 ..< numCourses where degree[i] == 0 {
        queue.append(i)
    }
    // 直到队列为空
    while !queue.isEmpty {
        // 取出第一个顶点
        let cur = queue.removeFirst()
        numCourses -= 1
        // 依次对邻接点的入度减一
        for next in adj[cur] {
            degree[next] -= 1
            // 如果入度为0则加入队列
            if degree[next] == 0 {
                queue.append(next)
            }
        }
    }
    return numCourses == 0
}
```

[986. Interval List Intersections](https://leetcode.com/problems/interval-list-intersections/)
``` swift
// 时间复杂度：O(m + n)
// 空间复杂度：O(1)
func intervalIntersection(_ A: [[Int]], _ B: [[Int]]) -> [[Int]] {
    var res = [[Int]]()
    var i = 0
    var j = 0
    while i < A.count && j < B.count {
        let sa = A[i][0]
        let ea = A[i][1]
        let sb = B[j][0]
        let eb = B[j][1]

        if sa <= eb && sb <= ea { // 只需要满足该条件即可
            res.append([max(sa, sb), min(ea, eb)])
        }
        if eb > ea {
            i += 1
        } else {
            j += 1
        }
    }
    return res
}
```

[230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)
``` swift
// 循环中序遍历
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func kthSmallest(_ root: TreeNode?, _ k: Int) -> Int {
    var stack = [TreeNode]()
    var count = k
    var cur = root
    while true {
        while let node = cur {
            stack.append(node)
            cur = node.left
        }
        cur = stack.removeLast()
        count -= 1
        if count == 0 {
            return cur!.val
        }
        cur = cur?.right
    }
    return 0
}

// 递归中序遍历
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func kthSmallest(_ root: TreeNode?, _ k: Int) -> Int {
    var res = 0
    var count = 0
    func inorder(_ root: TreeNode?) {
        guard let root = root else { return }
        inorder(root.left)
        count += 1
        if count == k {
            res = root.val
            return
        }
        inorder(root.right)
    }
    inorder(root)
    return res
}
```

[901. Online Stock Span](https://leetcode.com/problems/online-stock-span/)
``` swift
// 通过维护一个递减的栈来实现，利用 weight 字段来记录被 pop 出去的数的个数
// 时间复杂度：O(n)
// 空间复杂度：O(n)
class StockSpanner {
    
    var stack = [(Int, Int)]()

    func next(_ price: Int) -> Int {
        var w = 1
        while !stack.isEmpty && price >= stack.last!.0 {
            let last = stack.removeLast()
            w += last.1
        }
        stack.append((price, w))
        return w
    }
}
```

[567. Permutation in String](https://leetcode.com/problems/permutation-in-string/)
``` swift
// 时间复杂度：O(n)
// 空间复杂度：O(1)
func checkInclusion(_ s1: String, _ s2: String) -> Bool {
    guard s1.count <= s2.count else { return false }

    let s1 = s1.map { $0.asciiValue! - Character("a").asciiValue! }
    let s2 = s2.map { $0.asciiValue! - Character("a").asciiValue! }
    var dict = [Int](repeating: 0, count: 26)

    // start 和 end 记录窗口大小
    var start = 0
    var end = 0

    // 利用数组记录模式串各字符出现的次数
    for c in s1 {
        dict[Int(c)] += 1
    }

    var tmp: UInt8 = 0
    while end < s2.count {
        tmp = s2[end]

        // end 指向的字符在模式串内，减少对应字符的记录，并移动 end
        if dict[Int(tmp)] > 0 {
            dict[Int(tmp)] -= 1
            end += 1

            // 窗口的大小和模式串一致，匹配成功
            if end - start == s1.count {
                return true
            }
        // 窗口为 0 时，同时移动 start 和 end
        } else if start == end {
            start += 1
            end += 1
        // 否则移动 start，回收已匹配字符的记录
        } else {
            tmp = s2[start]
            dict[Int(tmp)] += 1
            start += 1
        }
    }

    return false
}
```

[328. Odd Even Linked List](https://leetcode.com/problems/odd-even-linked-list/)
``` swift
func oddEvenList(_ head: ListNode?) -> ListNode? {
    let odd = head
    var oddTail = odd
    let even = head?.next
    var evenTail = even
    var isOdd = true
    var cur = even?.next
    while cur != nil  {
        if isOdd {
            oddTail?.next = cur
            oddTail = cur
        } else {
            evenTail?.next = cur
            evenTail = cur
        }
        isOdd = !isOdd
        cur = cur?.next
    }
    evenTail?.next = nil
    oddTail?.next = even
    return odd
}
```

[918. Maximum Sum Circular Subarray](https://leetcode.com/problems/maximum-sum-circular-subarray/)
``` swift
// Kadane 算法
// 时间复杂度：O(logn)
// 空间复杂度：O(1)
//
// 这里有两种情况：
// 1、最大子序列在中间
// 2、最大子序列在两端
//
// 第一种情况可以用 Kadane 算法 直接求出，至于第二情况，我们可以通过原始序列之和减去最小子序列求出
func maxSubarraySumCircular(_ A: [Int]) -> Int {
    var total = 0
    var maxSum = -30000
    var curMax = 0
    var minSum = 30000
    var curMin = 0
    for n in A {
        curMax = max(n, curMax + n)
        maxSum = max(maxSum, curMax)
        curMin = min(n, curMin + n)
        minSum = min(minSum, curMin)
        total += n
    }
    return maxSum > 0 ? (max(maxSum, total - minSum)) : maxSum
}
```

[First Bad Version](https://leetcode.com/explore/challenge/card/may-leetcoding-challenge/534/week-1-may-1st-may-7th/3316/)
``` swift
// 二分
// 时间复杂度：O(logn)
func firstBadVersion(_ n: Int) -> Int {
    var l = 0
    var r = n
    while l < r {
        let mid = l + (r - l) / 2
        if !isBadVersion(mid) {
            l = mid + 1
        } else {
            r = mid
        }
    }
    return l
}
```

[124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)
``` swift
// DFS
// 时间复杂度：O(n)
func maxPathSum(_ root: TreeNode?) -> Int {
    var res = Int.min
    dfs(root, &res)
    return res
}

func dfs(_ root: TreeNode?, _ res: inout Int) -> Int {
    guard let root = root else { return 0 }
    
    let left = dfs(root.left, &res)
    let right = dfs(root.right, &res)
    let ret = max(root.val, max(root.val + left, root.val + right))
    
    res = max(res, max(max(root.val, ret), root.val + left + right))
    
    return ret
}
```

[Maximal Square](https://leetcode.com/problems/maximal-square/)
``` swift
// DP
// 时间复杂度：O(m * n)
func maximalSquare(_ matrix: [[Character]]) -> Int {
    guard matrix.count > 0 && matrix[0].count > 0 else { return 0 }
    
    let m = matrix.count
    let n = matrix[0].count
    var res = 0
    var last = 0
    var dp = [Int](repeating: 0, count: n)
    
    for (i, c) in matrix[0].enumerated() where c == Character("1") {
        dp[i] = 1
        res = 1
    }
    
    for i in 1 ..< m {
        last = dp[0]
        dp[0] = matrix[i][0] == Character("1") ? 1 : 0
        res = max(res, dp[0])
        for j in 1 ..< n {
            let tmp = dp[j]
            if matrix[i][j] == Character("1") {
                dp[j] = min(min(last, tmp), dp[j - 1]) + 1
                res = max(res, dp[j])
            } else {
                dp[j] = 0
            }
            last = tmp
        }
    }
    return res * res
}
```

[Check If a String Is a Valid Sequence from Root to Leaves Path in a Binary Tree](https://leetcode.com/explore/challenge/card/30-day-leetcoding-challenge/532/week-5/3315/)
``` swift
// DP
// 时间复杂度：O(n)
func isValidSequence(_ root: TreeNode?, _ arr: [Int]) -> Bool {
    return check(root, arr: arr, index: 0)
}

func check(_ root: TreeNode?, arr: [Int], index: Int) -> Bool {
    // 终止条件
    guard let root = root, index < arr.count, root.val == arr[index] else { return false }
    
    if index == arr.count - 1 && root.left == nil && root.right == nil {
        return true
    }

    // 本次递归所需的任务
    return check(root.left, arr: arr, index: index + 1) || check(root.right, arr: arr, index: index + 1)
}
```

[1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)
``` swift
// DP
// 时间复杂度：O(m * n)
func longestCommonSubsequence(_ text1: String, _ text2: String) -> Int {
    let s = Array(text1)
    let t = Array(text2)
    var dp = [[Int]](repeating: [Int](repeating: 0, count: t.count + 1), count: s.count + 1)
    for i in 1 ... s.count {
        for j in 1 ... t.count {
            if s[i - 1] == t[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1
            } else {
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            }
        }
    }
    return dp[s.count][t.count]
}
```

[33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
``` swift
func search(_ nums: [Int], _ target: Int) -> Int {
    var l = 0
    var h = nums.count - 1
    while l <= h {
        let mid = l + (h - l) / 2
        if nums[mid] == target {
            return mid
        } else if nums[mid] > nums[h] {
            if target > nums[mid] || target <= nums[h] {
                l = mid + 1
            } else {
                h = mid - 1
            }
        } else {
            if nums[mid] < target && target <= nums[h] {
                l = mid + 1
            } else {
                h = mid - 1
            }
        }
    }
    return -1
}
```

[64. Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/)
``` swift
// DP
// 时间复杂度：O(m * n)
func minPathSum(_ grid: [[Int]]) -> Int {
    guard grid.count > 0 && grid[0].count > 0 else { return 0 }
    
    let row = grid.count
    let column = grid[0].count
    var dp = grid[0]
    for j in 1 ..< column {
        dp[j] = dp[j - 1] + grid[0][j]
    }
    for i in 1 ..< row {
        dp[0] = dp[0] + grid[i][0]
        for j in 1 ..< column {
            dp[j] = min(dp[j], dp[j - 1]) + grid[i][j]
        }
    }

    return dp[column - 1]
}
```

[200. Number of Islands](https://leetcode.com/problems/number-of-islands/)
``` swift
// DFS
// 时间复杂度：O(n)
func numIslands(_ grid: [[Character]]) -> Int {
    var grid = grid
    var res = 0
    for i in 0 ..< grid.count {
        for j in 0 ..< grid[0].count where grid[i][j] == Character("1") {
            dfs(&grid, i, j)
            res += 1
        }
    }
    return res
}

func dfs(_ grid: inout [[Character]], _ i: Int, _ j: Int) {
    if i < 0
        || j < 0
        || i >= grid.count
        || j >= grid[0].count
        || grid[i][j] != Character("1") {
        return
    }
    grid[i][j] = Character("0")
    dfs(&grid, i, j + 1)
    dfs(&grid, i, j - 1)
    dfs(&grid, i + 1, j)
    dfs(&grid, i - 1, j)
}
```

[678. Valid Parenthesis String](https://leetcode.com/problems/valid-parenthesis-string/)
``` swift
// lo 和 hi 分别记录当前左括号可能的最少或最多个数
// 时间复杂度：O(n)
func checkValidString(_ s: String) -> Bool {
    var lo = 0
    var hi = 0
    for c in s {
        lo += c == Character("(") ? 1 : -1
        hi += c != Character(")") ? 1 : -1
        if hi < 0 {
            break
        }
        lo = max(lo, 0)
    }
    return lo == 0
}
```

[Perform String Shifts](https://leetcode.com/explore/challenge/card/30-day-leetcoding-challenge/529/week-2/3299/)
``` swift
// 因为左移和右移的操作可以相互抵消，所以只需要算出最后移动的方向和大小就可以了
// 时间复杂度：O(n)
func stringShift(_ s: String, _ shift: [[Int]]) -> String {
    let chars = Array(s)
    let change = shift.reduce(into: 0) { $0 += ($1[0] == 0 ? 1 : -1) * $1[1] } % s.count
    if change > 0 {
        return String(chars[change ..< s.count]) + chars[0 ..< change]
    } else if change < 0 {
        return String(chars[s.count + change ..< s.count]) + chars[0 ..< s.count + change]
    }
    return s
}
```

[525. Contiguous Array](https://leetcode.com/submissions/detail/324573139/)
``` swift
// 核心思路是一个值 count 标记当前遇到“1”和“0”的情况，
// 除了等于 0 外，当 count 再次和以前的值相等时，说明遇到了相同个数的“1”和“0”
// 这里用哈希表去记录，key 为 count 的值，value 为该值第一次出现时在 nums 的索引
// 时间复杂度：O(n)
func findMaxLength(_ nums: [Int]) -> Int {
    var res = 0
    var dict = [0: -1]
    var count = 0
    
    for (i, n) in nums.enumerated() {
        count += (n == 1 ? 1 : -1)
        if let v = dict[count] {
            res = max(res, i - v)
        } else {
            dict[count] = i
        }
    }
    
    return res
}
```

[99. Recover Binary Search Tree](https://leetcode.com/problems/recover-binary-search-tree/)
``` swift
// 通过中序遍历，找出不正确的两个节点，具体可能有以下两种情况：
// 1、两个连着的：.. < .. < .. A > B .. < ..
// 2、两个不连着：.. < A > X .. < Y > B .. < ..
// 解决方法是找到 A 跟 B，然后交换
// 时间复杂度：O(n)
func recoverTree(_ root: TreeNode?) {
    var cur = root
    var pre = TreeNode(Int.min)
    var pairs = [(TreeNode, TreeNode)]()
    var stack = [TreeNode]()
    while cur != nil || stack.count > 0 {
        while cur != nil {
            stack.append(cur!)
            cur = cur?.left
        }
        cur = stack.removeLast()
        if let val = cur?.val, val < pre.val {
            pairs.append((pre, cur!))
        }
        pre = cur!
        cur = cur?.right
    }
    if pairs.count == 1 {
        (pairs[0].0.val, pairs[0].1.val) = (pairs[0].1.val, pairs[0].0.val)
    } else if pairs.count == 2 {
        (pairs[0].0.val, pairs[1].1.val) = (pairs[1].1.val, pairs[0].0.val)
    }
}
```

[85. Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)
``` swift
// 解法和 Largest Rectangle in Histogram 一样
// 时间复杂度：O(n^2)
func maximalRectangle(_ matrix: [[Character]]) -> Int {
    guard matrix.count > 0 && matrix[0].count > 0 else { return 0 }
    var res = 0
    let row = matrix.count
    let col = matrix[0].count
    var heights = [Int](repeating: 0, count: col + 1)
    for i in 0 ..< row {
        var stack = [Int]()
        for j in 0 ..< heights.count {
            if j < col && matrix[i][j] == "1" {
                heights[j] += 1
            } else {
                heights[j] = 0
            }
            
            while stack.count > 0 && heights[j] < heights[stack.last!] {
                let h = stack.removeLast()
                let pre = stack.count > 0 ? stack.last! : -1
                res = max((j - pre - 1) * heights[h], res)
            }
            stack.append(j)
        }
    }
    return res
}
```

[49. Group Anagrams](https://leetcode.com/problems/group-anagrams/)
``` swift
// 关键是构造 key 的效率上，例如采用：hash += UInt64(pow(5.0, Double(value)))
// 时间复杂度：O(n)
func groupAnagrams(_ strs: [String]) -> [[String]] {
    let strs = strs.map { Array($0) }
    var dict = [String: [String]]()
    for chars in strs {
            let key = String(chars.sorted())
        if let _ = dict[key] {
            dict[key]?.append(String(chars))
        } else {
            dict[key] = [String(chars)]
        }
    }
    return dict.values.map{ $0 }
}
```

[Counting Elements](https://leetcode.com/explore/challenge/card/30-day-leetcoding-challenge/528/week-1/3289/)
``` swift
// 时间复杂度：O(n)
func countElements(_ arr: [Int]) -> Int {
    var res = 0
    let dict = arr.reduce(into: [:]) { $0[$1] = $0[$1, default: 0] + 1 }
    for k in dict.keys where dict[k - 1] != nil {
        res += dict[k - 1]!
    }
    return res
}
```

[84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
``` swift
// 栈法
// 对于每一个柱子，找到左边的比它短的i, 右边比他短的j，然后局部max就是(j - i - 1) * h
// 时间复杂度：O(n)
func largestRectangleArea(_ heights: [Int]) -> Int {
    var res = 0
    var i = 0
    var stack = [Int]()
    var heights = heights
    heights.append(0) // 关键
    
    while i < heights.count {
        while stack.count > 0 && heights[i] < heights[stack.last!] {
            let h = stack.removeLast()
            let pre = stack.count > 0 ? stack.last! : -1
            res = max((i - pre - 1) * heights[h], res)
        }
        
        stack.append(i)
        i += 1
    }
    
    return res
}
```

[76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
``` swift
// 双指针法
// 时间复杂度：O(n)
func minWindow(_ s: String, _ t: String) -> String {
    let s = Array(s)
    let t = Array(t) // Swift 处理字符串实在太慢了，不先转为数组一定会超时
    var dict = t.reduce(into: [:]) { $0[$1] = $0[$1, default: 0] + 1 }
    var minCount = s.count + 1
    var minLeft = 0
    var left = 0
    var matches = 0
    
    for (i, c) in s.enumerated() { // 用 enumerated 会比 while 方式更慢
        if let count = dict[c] {
            dict[c] = count - 1 // 可以为负数，表示已超出的重复个数
            if count > 0 {
                matches += 1
            }
            
            while matches == t.count {
                if i - left + 1 < minCount {
                    minCount = i - left + 1
                    minLeft = left
                }
                if let count = dict[s[left]] {
                    dict[s[left]] = count + 1
                    if count >= 0 {
                        matches -= 1 // 已排除掉超出的重复个数，因此减一
                    }
                }
                left += 1 // 不断左移
            }
        }
    }
    
    if minCount > s.count {
        return ""
    }
    return String(s[minLeft ..< minLeft + minCount])
}
```

[68. Text Justification](https://leetcode.com/problems/text-justification/)
``` swift
// 解法不难，需要注意的是细节
// 时间复杂度：O(n)
func fullJustify(_ words: [String], _ maxWidth: Int) -> [String] {
    var res = [String]()
    var curLength = 0
    var wordLength = 0
    var line = [String]()
    for w in words {
        if (curLength + w.count + (curLength > 0 ? 1 : 0)) <= maxWidth {
            line.append(w)
            wordLength += w.count
            curLength += w.count + (curLength > 0 ? 1 : 0)
        } else {
            let spaces = maxWidth - wordLength
            var space = maxWidth - curLength
            var extra = 0
            if line.count > 1 {
                space = spaces / (line.count - 1)
                extra = spaces % (line.count - 1)
            }
            
            var l = ""
            for i in 0 ..< line.count {
                l += line[i]
                if i != line.count - 1 {
                    l += String(repeating: " ", count: space)
                    if extra > 0 {
                        l += " "
                        extra -= 1
                    }
                }                
            }
            if l.count < maxWidth {
                l += String(repeating: " ", count: maxWidth - l.count)
            }
            res.append(l)
            
            line = [w]
            wordLength = w.count
            curLength = w.count
        }
    }
    if line.count > 0 {
        var l = ""
        for i in 0 ..< line.count {
            l += line[i]
            if i != line.count - 1 {
                l += " "
            }
        }
        if l.count < maxWidth {
            l += String(repeating: " ", count: maxWidth - l.count)
        }
        res.append(l)
    }
    
    return res
}
// 简化版
func fullJustify(_ words: [String], _ maxWidth: Int) -> [String] {
    var res = [String]()
    var wordLength = 0
    var line = [String]()
    for w in words {
        if wordLength + w.count + line.count > maxWidth {
            for i in 0 ..< maxWidth - wordLength {
                if line.count == 1 {
                    line[0] += " "
                } else {
                    // 通过 Round Robin 法添加空格
                    line[i % (line.count - 1)] += " "
                }
            }
            
            res.append(line.joined())
            line = [String]()
            wordLength = 0
        }
        
        line.append(w)
        wordLength += w.count
    }
    
    if line.count > 0 {
        var last = line.joined(separator: " ")
        last += String(repeating: " ", count: maxWidth - last.count)
        res.append(last)
    }
    
    return res
}
```

[154. Find Minimum in Rotated Sorted Array II](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii)
``` swift
// 递归解法
// 时间复杂度：O(n)
func findMin(_ nums: [Int]) -> Int {
    guard nums.count > 1 else { return nums.first ?? -1 }
    return find(nums, 0, nums.count - 1)
}

func find(_ nums: [Int], _ i: Int, _ j: Int) -> Int {
    if nums[i] < nums[j] {
        return nums[i]
    }
    
    if i == j {
        return nums[i]
    }
    
    let m = i + (j - i) / 2
    
    if nums[m] > nums[m + 1] {
        return nums[m + 1]
    }
    
    if nums[i] == nums[j] {
        return min(find(nums, i, m), find(nums, m + 1, j))
    }
    
    // m > j，那么最小值在右边
    if nums[m] > nums[j] {
        return find(nums, m + 1, j)
    }
    
    // 否则左边
    return find(nums, i, m)
}

// 迭代法
func findMin(_ nums: [Int]) -> Int {
    var l = 0
    var r = nums.count - 1
            
    while l <= r {
        let m = l + (r - l) / 2
        if nums[m] > nums[r] {
            l = m + 1
        } else if nums[m] < nums[r] {
            r = m
        } else {
            if r - 1 > 0 && nums[r - 1] > nums[r] { // 为了找出 pivot 的 index
                l = r
                break
            }
            r -= 1 // 遇到重复，逐步缩小右边
        }
    }
    
    return nums[l]
}
```

[283. Move Zeroes](https://leetcode.com/problems/move-zeroes/)
``` swift
// 双指针法
// 时间复杂度：O(logn)
func moveZeroes(_ nums: inout [Int]) {
    var slow = 0
    for i in 0 ..< nums.count {
        if nums[i] != 0 {
            nums.swapAt(slow, i)
            slow += 1
        }
    }
}
```

[153. Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)
``` swift
// 二分搜索
// 时间复杂度：O(logn)
func findMin(_ nums: [Int]) -> Int {
    var l = 0
    var r = nums.count - 1
    
    guard nums[l] > nums[r] else { return nums[l] }
    
    while l <= r {
        let m = l + (r - l) / 2
        if nums[m] > nums[l] && nums[m] > nums[r] {
            l = m
        } else if nums[l] > nums[m] {
            r = m
        } else {
            break
        }
    }
    
    return nums[r]
}

// 简化版本
func findMin(_ nums: [Int]) -> Int {
    var l = 0
    var r = nums.count - 1
    
    guard nums[l] > nums[r] else { return nums[l] }
    
    while l + 1 < r {
        let m = l + (r - l) / 2
        if nums[m] > nums[l] {
            l = m
        } else {
            r = m
        }
    }
    
    return nums[r]
}
```

[145. Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)
``` swift
// 迭代法遍历
// 时间复杂度：O(n)
func postorderTraversal(_ root: TreeNode?) -> [Int] {
    var res = [Int]()
    var stack = [TreeNode]()
    var cur = root
    var pre: TreeNode?
    
    while cur != nil || stack.count > 0 {
        // 递归遍历左节点
        while cur != nil {
            stack.append(cur!)
            cur = cur?.left
        }
        
        // 已经访问到最左节点
        cur = stack.last
        // 不存在右节点或右节点已经访问过时，访问跟节点
        if cur?.right == nil || cur?.right === pre {
            stack.removeLast()
            res.append(cur!.val)
            pre = cur
            cur = nil
        } else {
            cur = cur?.right
        }
    }
    
    return res
}
```

[65. Valid Number](https://leetcode.com/problems/valid-number/)
``` swift
// 用状态机逐个处理
// 时间复杂度：O(n)
func isNumber(_ s: String) -> Bool {
    let s = Array(s.trimmingCharacters(in: .whitespaces))
    var state = 0
    for c in s {
        if c == Character("+") || c == Character("-") {
            switch state {
            case 0:
                state = 1
            case 4:
                state = 6
            default:
                state = -1
                break
            }
        } else if c.isWholeNumber {
            switch state {
            case 0: fallthrough
            case 1: fallthrough
            case 2:
                state = 2
            case 3:
                state = 3
            case 4: fallthrough
            case 5: fallthrough
            case 6:
                state = 5
            case 7: fallthrough
            case 8:
                state = 8
            default:
                state = -1
                break
            }
        } else if c == Character(".") {
            switch state {
            case 0: fallthrough
            case 1:
                state = 7
            case 2:
                state = 3
            default:
                state = -1
                break
            }
        } else if c == Character("e") {
            switch state {
            case 2: fallthrough
            case 3: fallthrough
            case 8:
                state = 4
            default:
                state = -1
                break
            }
        } else {
            state = -1
            break
        }
    }
    
    return state == 2 || state == 3 || state == 5 || state == 8
}

// 正则解法
func isNumber(_ s: String) -> Bool {
    let s = s.trimmingCharacters(in: .whitespaces)
    if let res = s.range(of: "[-+]?(\\d+\\.?|\\.\\d+)\\d*(e[-+]?\\d+)?", options: .regularExpression) {
        return s[res] == s
    }
    return false
}
```
[51. N-Queens](https://leetcode.com/problems/n-queens/)
``` swift
// 时间复杂度：O(n^2)
func solveNQueens(_ n: Int) -> [[String]] {
    var positions = [Int]() // 记录每行 Q 的位置
    var res = [[Int]]()
    solve(&positions, &res, n)
    return translateToBoard(res, n)
}

func solve(_ positions: inout [Int], _ res: inout [[Int]], _ n: Int) {
    // 已经算到最后一行，满足条件，保存
    if n == positions.count {
        let copy = positions
        res.append(copy)
        return
    }
    
    for j in 0 ..< n where isValid(positions, j) {
        positions.append(j)
        solve(&positions, &res, n)
        positions.removeLast()
    }
}

func isValid(_ pos: [Int], _ col: Int) -> Bool {
    guard pos.count > 0 else { return true }
    
    for i in 0 ..< pos.count {
        // 上方
        if pos[i] == col {
            return false
        }
        // 右上方
        if pos[i] == col + pos.count - i {
            return false
        }
        // 左上方
        if pos[i] == col - pos.count + i {
            return false
        }
    }
    
    return true
}

func translateToBoard(_ res: [[Int]], _ n: Int) -> [[String]] {
    var boards = [[String]]()
    for pos in res {
        var chars = [[Character]](repeating: [Character](repeating: ".", count: n), count: n)
        for (i, p) in pos.enumerated() {
            chars[i][p] = Character("Q")
        }
        boards.append(chars.map{ String($0) })
    }
    return boards
}
```

[136. Single Number](https://leetcode.com/problems/single-number/)
``` swift
// 通过异或消除重复数
// 时间复杂度：O(n)
func singleNumber(_ nums: [Int]) -> Int {
    var n = 0
    for i in nums {
        n ^= i
    }
    return n
}
```

[30. Substring with Concatenation of All Words](https://leetcode.com/problems/substring-with-concatenation-of-all-words/)
``` swift
func findSubstring(_ s: String, _ words: [String]) -> [Int] {
    guard s.count > 0
        && words.count > 0
        && words[0].count > 0
        && s.count >= words.count * words[0].count else {
            return []
    }
    
    // 通过哈希表去除排序问题
    let dict: [String: Int] = words.reduce(into: [:]) { $0[$1] = $0[$1, default: 0] + 1 }
    let size = words[0].count
    let length = words.count * size
    var res = [Int]()
    let chars = Array(s)
    
    for i in 0 ..< size {
        var start = i
        
        while start + length <= s.count {
            var copy = [String: Int]()
            var end = start + length
            var matches = 0
            
            while end > start {
                // swift 的 substring 太慢，用数组替代
                let word = String(chars[end - size ..< end])
                end -= size
                
                let count = copy[word, default: 0] + 1
                if count > dict[word] ?? 0 {
                    break
                }
                
                matches += 1
                copy[word] = count
            }
            
            // 利用滑动窗口，减少运行时间
            if matches == words.count {
                res.append(start)
                start += size
            } else {
                start = max(end + size, start + size)
            }
        }
    }
    
    return res
}
```

[37. Sudoku Solver](https://leetcode.com/problems/sudoku-solver/)
``` swift
// 回溯
// 时间复杂度：O(n^3)
func solveSudoku(_ board: inout [[Character]]) {
    guard board.count > 0 && board[0].count > 0 else { return }

    _ = solve(&board)
}

func solve(_ board: inout [[Character]]) -> Bool {
    for i in 0 ..< board.count {
        for j in 0 ..< board[0].count {
            if board[i][j] == Character(".") {
                for k in 1 ... 9 where isValid(board, i, j, Character("\(k)")) {
                    board[i][j] = Character("\(k)")
                    if solve(&board) {
                        return true
                    }
                    board[i][j] = Character(".")   
                }
                return false
            }
        }
    }

    return true
}

func isValid(_ board: [[Character]], _ i: Int, _ j: Int, _ k: Character) -> Bool {
    for q in 0 ..< 9 {
        if board[i][q] != Character(".") && board[i][q] == k {
            return false
        }
        if board[q][j] != Character(".") && board[q][j] == k {
            return false
        }
        let m = 3 * (i / 3) + q / 3
        let n = 3 * (j / 3) + q % 3
        if board[m][n] != Character(".") && board[m][n] == k {
            return false
        }
    }
    return true
}
```

[72. Edit Distance](https://leetcode.com/problems/edit-distance/)
``` swift
// DP
func minDistance(_ word1: String, _ word2: String) -> Int {
    guard word1.count > 0 && word2.count > 0 else { return max(word1.count, word2.count) }

    var word1 = Array(word1)
    var word2 = Array(word2)
    if word2.count > word1.count {
        (word2, word1) = (word1, word2)
    }

    var dp = [Int](repeating: 0, count: word2.count + 1)
    var old = 0
    var tmp = 0

    for j in 0 ... word2.count {
        dp[j] = j
    }

    for i in 1 ... word1.count {
        old = i - 1
        dp[0] = i
        for j in 1 ... word2.count {
            tmp = dp[j]
            if word1[i - 1] == word2[j - 1] {
                dp[j] = old
            } else {
                dp[j] = min(dp[j] + 1, min(dp[j - 1] + 1, old + 1))
            }
            old = tmp
        }
    }

    return dp[word2.count]
}
```
[57. Insert Interval](https://leetcode.com/problems/insert-interval/)
``` swift
func insert(_ intervals: [[Int]], _ newInterval: [Int]) -> [[Int]] {
    var start = [[Int]]()
    var end = [[Int]]()
    var midStart = newInterval[0]
    var midEnd = newInterval[1]

    for i in intervals {
        if i[1] < midStart {
            start.append(i)
        } else if i[0] > midEnd {
            end.append(i)
        } else {
            midStart = min(midStart, i[0])
            midEnd = max(midEnd, i[1])
        }
    }

    return start + [[midStart, midEnd]] + end
}
```

[44. Wildcard Matching](https://leetcode.com/problems/wildcard-matching/)
``` swift
// O(n) 解法
func isMatch(_ s: String, _ p: String) -> Bool {
    let s = Array(s)
    let p = Array(p)
    var i = 0
    var j = 0
    var start = -1
    var match = 0
    while i < s.count {
        //advancing both pointers when (both characters match) or ('?' found in pattern)
        //note that *p will not advance beyond its length
        if j < p.count && (s[i] == p[j] || p[j] == Character("?")) {
            i += 1
            j += 1
        }
        // * found in pattern, track index of *, only advancing pattern pointer
        else if j < p.count && p[j] == Character("*") {
            start = j
            match = i
            j += 1
        }
        //current characters didn't match, last pattern pointer was *, current pattern pointer is not *
        //only advancing pattern pointer
        else if start != -1 {
            j = start + 1
            match += 1
            i = match
        }
        //current pattern pointer is not star, last patter pointer was not *
        //characters do not match
        else {
            return false
        }
    }
    
    //check for remaining characters in pattern
    while j < p.count && p[j] == Character("*") {
        j += 1
    }
    
    return j == p.count
}

// DP 的 bottom up 解法
func isMatch(_ s: String, _ p: String) -> Bool {
    let s = Array(s)
    let p = Array(p)
    var dp = [[Bool]](repeating: [Bool](repeating: false, count: p.count + 1), count: s.count + 1)
    
    dp[s.count][p.count] = true
    
    for j in stride(from: p.count - 1, to: -1, by: -1) {
        if p[j] == Character("*") {
            dp[s.count][j] = true
        } else {
            break
        }
    }
    
    for i in stride(from: s.count - 1, to: -1, by: -1) {
        for j in stride(from: p.count - 1, to: -1, by: -1) {
            if s[i] == p[j] || p[j] == Character("?") {
                dp[i][j] = dp[i + 1][j + 1]
            } else if p[j] == Character("*") {
                dp[i][j] = dp[i + 1][j] /* 匹配 0 个 */ || dp[i][j + 1] /* 匹配多个 */
            }
        }
    }
    
    return dp[0][0]
}

// 普通 DP 解法
var memo: [[Bool?]]?
func isMatch(_ s: String, _ p: String) -> Bool {
    memo = [[Bool?]](repeating: [Bool?](repeating: nil, count: p.count + 1), count: s.count + 1)
    let res = dp(0, 0, Array(s), Array(p))
    return res
}
func dp(_ i: Int, _ j: Int, _ s: [Character], _ p: [Character]) -> Bool {
    if let ans = memo?[i][j] {
        return ans
    }
       
    var ans = false
    if j == p.count {
        ans = i == s.count
    } else {
        if i < s.count && (s[i] == p[j] || p[j] == Character("?")) {
            ans = dp(i + 1, j + 1, s, p)
        } else if p[j] == Character("*") {
            if i < s.count {
                ans =  dp(i, j + 1, s, p) /* 匹配 0 个 */
                    || dp(i + 1, j, s, p) /* 匹配多个 */
            } else {
                ans = dp(i, j + 1, s, p)
            }
        }
    }
    memo?[i][j] = ans
    return ans
}
```

[42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)
``` swift
// 双向指针解法
// 类似于把水向中间逼近
func trap(_ height: [Int]) -> Int {
    var res = 0
    var left = 0
    var right = height.count - 1
    var maxLeft = 0
    var maxRight = 0
    while left <= right {
        if height[left] <= height[right] {
            if height[left] > maxLeft {
                maxLeft = height[left]
            } else {
                res += maxLeft - height[left]
            }
            left += 1
        } else {
            if height[right] > maxRight {
                maxRight = height[right]
            } else {
                res += maxRight - height[right]
            }
            right -= 1
        }
    }
    
    return res
}

// 栈解法
func trap(_ height: [Int]) -> Int {
    var res = 0
    var stack = [Int]()
    var i = 0
    while i < height.count {
        if stack.count == 0 || height[i] <= height[stack.last!] { // 保留下降的 index
            stack.append(i)
            i += 1
        } else {
            let pre = stack.removeLast() // 当前位置最低
            if stack.count > 0 { // 处理 edge case
                let minHeight = min(height[i], height[stack.last!])
                res += (minHeight - height[pre]) * (i - stack.last! - 1) // 左右 index 的差即为当前区间水量
            }
        }
    }
    return res
}
```
[44. Wildcard Matching](https://leetcode.com/problems/wildcard-matching/)
``` swift
// DP 做法
var memo: [[Bool?]]?

func isMatch(_ s: String, _ p: String) -> Bool {
    memo = [[Bool?]](repeating: [Bool?](repeating: nil, count: p.count + 1), count: s.count + 1)
    let res = dp(0, 0, Array(s), Array(p))
    return res
}

func dp(_ i: Int, _ j: Int, _ s: [Character], _ p: [Character]) -> Bool {
    if let ans = memo?[i][j] {
        return ans
    }
       
    var ans = false
    if j == p.count {
        ans = i == s.count
    } else {
        if i < s.count && (s[i] == p[j] || p[j] == Character("?")) {
            ans = dp(i + 1, j + 1, s, p)
        } else if p[j] == Character("*") {
            if i < s.count {
                ans =  dp(i, j + 1, s, p) /* 匹配 0 个 */ 
                    || dp(i + 1, j + 1, s, p) /* 匹配 1 个 */ 
                    || dp(i + 1, j, s, p) /* 匹配多个 */
            } else {
                ans = dp(i, j + 1, s, p)
            }
        }
    }
    memo?[i][j] = ans
    return ans
}
```

[45. Jump Game II](https://leetcode.com/problems/jump-game-ii/)
``` swift
// BFS
func jump(_ nums: [Int]) -> Int {
    guard nums.count > 0 else { return 0 }
    
    var jump = 0
    var end = 0
    var maxDist = 0
    for i in 0 ..< nums.count - 1 {
        maxDist = max(maxDist, nums[i] + i)
        if i == end {
            jump += 1
            end = maxDist
        }
    }
    return jump
}
```

[41. First Missing Positive](https://leetcode.com/problems/first-missing-positive/)
``` swift
// 用查表的思路，把数放到对应位置：5 => nums[4]
func firstMissingPositive(_ nums: [Int]) -> Int {
    var nums = nums
    var n = nums.count

    for i in 0 ..< n {
        while nums[i] > 0 && nums[i] < n && nums[i] != nums[nums[i] - 1] {
            (nums[i], nums[nums[i] - 1]) = (nums[nums[i] - 1], nums[i])
        }
    }

    for i in 0 ..< n where nums[i] != i + 1 {
        return i + 1
    }

    return n + 1
}
```

[32. Longest Valid Parentheses](https://leetcode.com/problems/longest-valid-parentheses/)
``` swift
// 使用 DP 的解法
func longestValidParentheses(_ s: String) -> Int {
    guard s.count > 1 else { return 0 }
    
    let s = Array(s)
    let n = s.count
    let left = Character("(")
    let right = Character(")")
    var dp = [Int](repeating: 0, count: n)
    var longest = 0
    for i in 1 ..< n {
        let last = i - dp[i - 1] - 1
        if s[i] == right && last >= 0 && s[last] == left {
            let previous = i - dp[i - 1] - 2
            dp[i] = dp[i - 1] + 2 + (previous >= 0 ? dp[previous] : 0)
            longest = max(longest, dp[i])
        }
    }
    
    return longest
}

// 使用栈的解法
func longestValidParentheses(_ s: String) -> Int {
    var stack = [Int]()
    let left = Character("(")
    let s = Array(s)
    for (idx, c) in s.enumerated() {
        if c == left {
            stack.append(idx)
        } else {
            if let last = stack.last, s[last] == left { // 找到匹配，从栈中去除
                stack.removeLast()
            } else {
                stack.append(idx)
            }
        }
    }
    if stack.count == 0 {
        return s.count
    }

    var longest = 0
    var n = s.count
    while stack.count != 0 {
        let b = stack.last!
        stack.removeLast()
        longest = max(longest, n - b - 1)
        n = b
    }
    return max(longest, n)
}
```

[25. Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)
``` swift
func reverseKGroup(_ head: ListNode?, _ k: Int) -> ListNode? {
    // 对 k 个进行分组
    var tmp = head
    var i = 1
    while tmp != nil && i < k {
        tmp = tmp?.next
        i += 1
    }
    if tmp == nil {
        return head
    }
    
    // 断开
    let rest = tmp?.next
    tmp?.next = nil
    
    // 逆转
    let newHead = reverse(head)
    // 递归处理下一部分
    let nextPart = reverseKGroup(rest, k)
    // 重新连接
    head?.next = nextPart
    
    return newHead
}

func reverse(_ head: ListNode?) -> ListNode? {
    guard head?.next != nil else { return head }
    let newHead = reverse(head?.next)
    head?.next?.next = head
    head?.next = nil
    return newHead
}
```

[23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)
``` swift
// 通过分治的方式，两两进行排序合并
func mergeKLists(_ lists: [ListNode?]) -> ListNode? {
    guard lists.count > 0 else { return nil }

    let count = lists.count
    var l = lists
    var interval = 1
    while interval < count {
        var i = 0
        while i < count - interval {
            l[i] = merge2Lists(l[i], l[i + interval])
            i += interval * 2
        }
        interval *= 2
    }

    return lists[0]
}

func merge2Lists(_ left: ListNode?, _ right: ListNode?) -> ListNode? {
    var l = left
    var r = right
    let head: ListNode? = ListNode(0)
    var p = head
    while let lv = l?.val, let rv = r?.val {
        if lv <= rv {
            p?.next = l
            l = l?.next
        } else {
            p?.next = r
            r = r?.next
        }
        p = p?.next
    }
    if l != nil {
        p?.next = l
    }
    if r != nil {
        p?.next = r
    }
    return head?.next
}
```
[10. Regular Expression Matching](https://leetcode.com/problems/regular-expression-matching/)
``` swift
// 这里可以用动态规划解决，
// 除去“*”号的话，要匹配的其实就只是：s[i] == p[j] || p[i] == '.'，
// 而包含“*”的模式，那么我们可以选择忽略这个模式（对应 0 个该字符），后者用下一个字符去匹配（对应 1 或多个该字符），
// 具体实现方式也有两种：Bottom-Up 和 Top-Down 
//
// Bottom-Up
func isMatch(_ s: String, _ p: String) -> Bool {
    let dot = Character(".")
    let star = Character("*")
    let s = Array(s)
    let p = Array(p)
    var dp = [[Bool]](repeating: [Bool](repeating: false, count: p.count + 1), count: s.count + 1)
    dp[s.count][p.count] = true

    for i in stride(from: s.count, to: -1, by: -1) {
        for j in stride(from: p.count - 1, to: -1, by: -1) {
            let prefixMatch = (i < s.count && (p[j] == s[i] || p[j] == dot))
            if j + 1 < p.count && p[j + 1] == star {
                dp[i][j] = dp[i][j + 2] || (prefixMatch && dp[i + 1][j]) // 如果下一个是"*"号，那么我们可以忽略当前这个模式，后者用下一个字符去匹配
            } else {
                dp[i][j] = prefixMatch && dp[i + 1][j + 1]
            }
        }
    }

    return dp[0][0]
}

// Top-Down
var memo: [[Bool?]]?
func isMatch(_ s: String, _ p: String) -> Bool {
    memo = [[Bool?]](repeating: [Bool?](repeating: nil, count: p.count + 1), count: s.count + 1)
    return dp(0, j: 0, s: Array(s), p: Array(p))
}

func dp(_ i: Int, j: Int, s: [Character], p: [Character]) -> Bool {
    if let res = memo?[i][j] {
        return res
    }
    
    var ans = false
    if j == p.count {
        ans = i == s.count
    } else {
        let prefixMatch = (i < s.count && (s[i] == p[j] || p[j] == Character(".")))
        if j + 1 < p.count && p[j + 1] == Character("*") {
            ans = dp(i, j: j + 2, s: s, p: p) || (prefixMatch && dp(i + 1, j: j, s: s, p: p))
        } else {
            ans = prefixMatch && dp(i + 1, j: j + 1, s: s, p: p)
        }
    }
    
    memo![i][j] = ans
    return ans
}
```

[4. Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/)
``` swift
// 首先我们先设定 i，j 把 A，B 各分成左右两部分，同时我们设定 m(len(A)) < n(len(B))
// 那么需要满足：
// 1、j = (m + n + 1) / 2 - i
// 2、B[j - 1] <= A[i]，A[i - 1] <= B[j]
// 这里使用二分搜索法，寻找在 [0, m] 范围内，满足条件的 i 值
// 搜索的过程中会遇到 3 种情况：
// 1、(j == 0 || i == m || B[j - 1] <= A[i]) && (i == 0 || j == n || A[i - 1] <= B[j])
//    满足条件，停止搜索
// 2、j > 0 && i < m && B[j - 1] > A[i]
//    i 太小
// 3、i > 0 && j < n && A[i - 1] > B[j]
//    i 太大
func findMedianSortedArrays(_ nums1: [Int], _ nums2: [Int]) -> Double {
    var A = nums1
    var B = nums2
    if A.count > B.count {
        (A, B) = (B, A)
    }
    let m = A.count
    let n = B.count
    let halfLen = (m + n + 1) / 2
    var imin = 0
    var imax = m
    while imin <= imax {
        let i = (imin + imax) / 2
        let j = halfLen - i
        if i < imax && B[j - 1] > A[i] {
            imin = i + 1
        } else if i > imin && A[i - 1] > B[j] {
            imax = i - 1
        } else {
            var maxLeft = 0
            if i == 0 {
                maxLeft = B[j - 1]
            } else if j == 0 {
                maxLeft = A[i - 1]
            } else {
                maxLeft = max(A[i - 1], B[j - 1])
            }
            
            if (m + n) % 2 == 1 {
                return Double(maxLeft)
            }
            
            var minRight = 0
            if i == m {
                minRight = B[j]
            } else if j == n {
                minRight = A[i]
            } else {
                minRight = min(A[i], B[j])
            }
            
            return Double(minRight + maxLeft) / 2
        }
    }
    return 0
}
```

[1014. Best Sightseeing Pair](https://leetcode.com/problems/best-sightseeing-pair/)
``` swift
func maxScoreSightseeingPair(_ A: [Int]) -> Int {
    var maxS = A[0] - 1
    var res = 0
    for i in 1 ..< A.count {
        res = max(res, maxS + A[i])
        maxS = max(maxS, A[i])
        maxS -= 1
    }
    return res
}
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
``` swift
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
``` swift
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