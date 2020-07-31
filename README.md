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