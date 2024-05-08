from typing import List

"""动态规划5步曲
1.确定dp数组以及下标的含义
2.确定递推公式
3.dp数组如何初始化
4.确定遍历顺序
5.举例推导dp数组
"""


"""leetcode70 爬楼梯
假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

 

示例 1：

输入：n = 2
输出：2
解释：有两种方法可以爬到楼顶。
1. 1 阶 + 1 阶
2. 2 阶
示例 2：

输入：n = 3
输出：3
解释：有三种方法可以爬到楼顶。
1. 1 阶 + 1 阶 + 1 阶
2. 1 阶 + 2 阶
3. 2 阶 + 1 阶
"""
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def climb_stairs(n: int) -> int:
    if n <= 1:
        return n

    # dp[i]： 爬到第i层楼梯，有dp[i]种方法
    dp = [0] * (n + 1)  # dp数组初始化
    dp[1] = 1
    dp[2] = 2

    for i in range(n + 1):  # 遍历顺序
        dp = dp[i - 1] + dp[i - 2]

    print("dp数组：", dp)  # 5.举例check dp数组
    return dp[n]


"""leetcode746 使用最小花费爬楼梯
给你一个整数数组 cost ，其中 cost[i] 是从楼梯第 i 个台阶向上爬需要支付的费用。一旦你支付此费用，即可选择向上爬一个或者两个台阶。

你可以选择从下标为 0 或下标为 1 的台阶开始爬楼梯。

请你计算并返回达到楼梯顶部的最低花费。


示例 1：

输入：cost = [10,15,20]
输出：15
解释：你将从下标为 1 的台阶开始。
- 支付 15 ，向上爬两个台阶，到达楼梯顶部。
总花费为 15 。
示例 2：

输入：cost = [1,100,1,1,1,100,1,1,100,1]
输出：6
解释：你将从下标为 0 的台阶开始。
- 支付 1 ，向上爬两个台阶，到达下标为 2 的台阶。
- 支付 1 ，向上爬两个台阶，到达下标为 4 的台阶。
- 支付 1 ，向上爬两个台阶，到达下标为 6 的台阶。
- 支付 1 ，向上爬一个台阶，到达下标为 7 的台阶。
- 支付 1 ，向上爬两个台阶，到达下标为 9 的台阶。
- 支付 1 ，向上爬一个台阶，到达楼梯顶部。
总花费为 6 。
"""


def min_cost_climb_stairs(cost: List[int]) -> int:
    # dp数组初始化
    dp = [float("inf") for i in range(len(cost) + 1)]
    dp[0] = 0
    dp[1] = 0

    for i in range(2, len(cost) + 1):  # 遍历顺序
        dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])  # 状态转移方程

    print("dp数组:", dp)

    return dp[len(cost)]


"""leetcode62 不同路径
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
问总共有多少条不同的路径？

输入：m = 3, n = 7
输出：28

"""


def unique_path(m: int, n: int) -> int:
    if m <= 0 or n <= 0:
        return
    # dp数组初始化
    dp = [[0] * n for _ in range(m + 1)]
    for i in range(m):
        dp[i][0] = 1
    for j in range(n):
        dp[0][j] = 1

    for i in range(m):  # 遍历顺序
        for j in range(n):  # 遍历顺序
            dp[i][j] = dp[i - 1] + dp[j - 1]

    print("dp数组校验：", dp)  # dp数组校验

    return dp[m - 1][n - 1]


"""leetcode63 不同路径 II
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

输入：obstacleGrid = [[0,1],[0,0]]
输出：1
"""


def unique_path_with_obstacle(obstacle_grid: List) -> int:
    m = len(obstacle_grid)
    n = len(obstacle_grid[0])

    if obstacle_grid[m - 1][n - 1] == 1 or obstacle_grid[0][0] == 1:
        return 0

    # dp 数组初始化
    dp = [[0] * n for _ in range(m)]
    for i in range(m):
        if obstacle_grid[i][0] == 0:
            dp[i][0] = 1
        else:  # 初始化后面路径为0
            break
    for j in range(n):
        if obstacle_grid[0][j] == 0:
            dp[0][j] = 1
        else:
            break

    for i in range(1, m):  # 遍历顺序
        for j in range(1, n):
            if obstacle_grid[i][j] == 1:  # 遇到障碍物时结束当前路径
                continue
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

    return dp[m - 1][n - 1]


"""leetcode343 整数拆分
给定一个正整数 n，将其拆分为至少两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。

示例 1:

输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1。
示例 2:

输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36。
说明: 你可以假设 n 不小于 2 且不大于 58。
"""


def integerBreak(n: int) -> int:
    """整数拆分,给定一个正整数 n，将其拆分为至少两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积

    Args:
        n (int): _description_

    Returns:
        int: _description_
    """
    # dp[i]: 分拆数字i，可以得到的最大乘积为dp[i]
    dp = [0] * (n + 1)
    dp[2] = 1
    # dp[3] = 2

    for i in range(3, n + 1):
        for j in range(1, i):
            dp[i] = max(dp[i], dp[i - j] * j, j * (i - j))

    return dp[n]


"""leetcode96 不同的二叉搜索树

给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种
输入：3
输出：5

输入：5
输出: 42
"""


def nums_of_binary_search_tree(n: int):

    # dp dp[i] ： 1到i为节点组成的二叉搜索树的个数为dp[i]
    dp = [0] * (n + 1)  # 数组初始化
    dp[0] = 1  # base case

    for i in range(1, n + 1):  # 遍历顺序
        for j in range(1, i + 1):  # 计算逻辑
            dp[i] += dp[j - 1] * dp[i - j]  # 计算逻辑 状态转移方程

    print("dp数组：", dp)
    return dp[n]


class Solution:
    def nums_of_binary_search_tree(self, n: int) -> int:
        self.memo = [[0] * (n + 1) for _ in range(n + 1)]  # 备忘录
        return self.count(1, n)

    def count(self, lo, hi):
        if lo > hi:
            return 1

        # 查备忘录
        if self.memo[lo][hi] != 0:
            return self.memo[lo][hi]

        res = 0
        for i in range(lo, hi + 1):
            left = self.count(lo, i - 1)
            right = self.count(i + 1, hi)
            res += left * right
        self.memo[lo][hi] = res
        return res


"""leetcode72 编辑距离

思路：解决两个字符串的动态规划问题，一般都是用两个指针 i, j 分别指向两个字符串的最后，然后一步步往前移动，缩小问题的规模。

给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。

你可以对一个单词进行如下三种操作：

插入一个字符
删除一个字符
替换一个字符
示例 1：

输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
"""
class Solution:
    def __init__(self) -> None:
        self.memo = []

    def minDistance(self, s1: str, s2: str):
        m = len(s1)
        n = len(s2)
        self.memo = [[-1] * n for _ in range(m + 1)]

        return self.dp(s1, m - 1, s2, n - 1)

    def dp(self, s1: str, i: int, s2: str, j: int):
        """dp函数+memo备忘录解法

        Args:
            s1 (str): _description_
            i (int): _description_
            s2 (str): _description_
            j (int): _description_

        Returns:
            _type_: _description_
        """
        if i == -1:
            return j + 1
        if j == -1:
            return i + 1

        if self.memo[i][j] != -1:
            return self.memo[i][j]

        if s1[i] == s2[j]:
            self.memo[i][j] = self.dp(s1, i - 1, s2, j - 1)
        else:
            self.memo[i][j] = min(
                self.dp(s1, i, s2, j - 1) + 1,  # 插入
                self.dp(s1, i - 1, s2, j) + 1,  # 删除
                self.dp(s1, i - 1, s2, j - 1) + 1,  # 替换
            )
        return self.memo


def minDistance(s1: str, s2: str):
    """解决两个字符串的动态规划问题，一般都是用两个指针 i, j 分别指向两个字符串的最后或开始，然后一步步往前移动，缩小问题的规模。
    使用动态规划算法，尽量先把dp数组初始化数组画出来
    Args:
        s1 (str): _description_
        s2 (str): _description_

    Returns:
        _type_: _description_
    """
    m = len(s1)
    n = len(s2)
    dp = [[0] * n + 1 for _ in range(m + 1)]  # dp数组初始化

    # base case
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    print("dp数组base case：", dp)

    for i in range(1, m + 1):  # 遍历顺序
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:  # 状态转移方程
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # 状态转移方程
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
    return dp[m][n]


"""leetcode300 最长递增子序列
"""
def length_of_LIS(nums: List[int]) -> int:

    # dp数组定义：dp[i] 表示以 nums[i] 这个数结尾的最长递增子序列的长度。
    dp = [1] * len(nums)  # dp[0]=1

    for i in range(1, len(nums)):  # 遍历顺序
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[i - 1] + 1)  # 状态转移方程
    res = 0
    for i in range(len(dp)):  # 遍历dp数组求最大值
        res = max(res, dp[i])
    # res = max(dp)
    return res


"""leetcode53 最大子数组和
输入：nums = [-3,1,3,-1,2,-4,2]，
输出：5
解释：因为最大子数组 [1,3,-1,2] 的和为 5。
"""
def max_sub_array(nums: List[int]) -> int:
    n = len(nums)
    if n == 0:
        return 0
    # dp数组定义：以 nums[i] 为结尾的「最大子数组和」为 dp[i]。
    dp = [0] * n
    dp[0] = nums[0]
    for i in range(1, n):  # 遍历顺序
        dp[i] = max(nums[i], nums[i] + dp[i - 1])  # 状态转移方程

    res = float("-inf")
    for i in range(n):  # 遍历dp数组获取最大值
        res = max(res, dp[i])

    return res

"""leetcode53 最大子数组和
输入：nums = [-3,1,3,-1,2,-4,2]，
算法返回 5，因为最大子数组 [1,3,-1,2] 的和为 5
"""
def max_sub_array(nums:List[int]) -> int:
    if not nums:
        return
    # dp：dp[i] 以nums[i]为结尾的最大子数组和为dp[i]
    dp = [0] * len(nums) 
    dp[0] = nums[0]
    for i in range(1,len(nums)):
        dp[i] = max(dp[i-1]+nums[i],nums[i])
    print("dp数组：",dp)

    res = float("-inf")
    for i in range(len(dp)):
        res = max(res,dp[i])
    return res

# 前缀和技巧解题
def maxSubArray(nums: List[int]) -> int:
    """以 nums[i] 为结尾的最大子数组之和是多少？其实就是 preSum[i+1] - min(preSum[0..i])
    """
    n = len(nums)
    preSum = [0] * (n + 1)
    preSum[0] = 0
    # 构造 nums 的前缀和数组
    for i in range(1, n + 1):
        preSum[i] = preSum[i - 1] + nums[i - 1]

    res = float('-inf')
    minVal = float('inf')
    for i in range(n):
        # 维护 minVal 是 preSum[0..i] 的最小值
        minVal = min(minVal, preSum[i])
        # 以 nums[i] 结尾的最大子数组和就是 preSum[i+1] - min(preSum[0..i])
        res = max(res, preSum[i + 1] - minVal)
    return res


"""leetcode1143 最长公共子序列
输入 s1 = "zabcde", s2 = "acez"，它俩的最长公共子序列是 lcs = "ace"，长度为 3
思路：对于两个字符串求子序列的问题，都是用两个指针 i 和 j 分别在两个字符串上移动，大概率是动态规划思路。
"""
class Solution:
    def __init__(self) -> None:
        self.memo = []
    def longestCommonSubsequence(self,s1:str,s2:str):
        m = len(s1)
        n = len(s2)
        self.memo = [[-1] * n for _ in range(m)]
        return self.dp(s1,0,s2,0)


    def dp(self,s1:str,i:int,s2:str,j:int):
        """dp 函数的定义是：dp(s1, i, s2, j) 计算 s1[i..] 和 s2[j..] 的最长公共子序列长度。
        接下来，咱不要看 s1 和 s2 两个字符串，而是要具体到每一个字符，思考每个字符该做什么
        Args:
            s1 (str): _description_
            i (int): _description_
            s2 (str): _description_
            j (int): _description_
        """
        if i == len(s1) or j == len(s2): # base case
            return 0
        if self.memo[i][j] != -1:
            return self.memo[i][j]
        
        if s1[i] == s2[j]:
            self.memo[i][j] = 1 + self.dp(s1,i+1,s2,j+1)
        else:
            self.memo[i][j] =  max(
                self.dp(s1,i+1,s2,j),
                self.dp(s1,i,s2,j+1),
                self.dp(s1,i+1,s2,j+1)
            )
        return self.memo[i][j]
    
def longestCommonSubsequence(s1:str,s2:str):
    m = len(s1)
    n = len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)] # base case
    for i in range(1,m+1): # 遍历顺序
        for j in range(1,n+1):
            if s1[i] == s2[j]:
                dp[i][j] = 1 + dp[i-1][j-1] # 状态转移方程
            else:
                dp[i][j] = max(dp[i-1][j],dp[i][j-1])  # 状态转移方程
    return dp[m][n]
    

"""leetcode 583两个字符串删除操作
输入：s1 = "sea" s2 = "eat"，
输出： 2
解释：第一步将 "sea" 变为 "ea" ，第二步将 "eat" 变为 "ea"
"""
def min_distance(s1:str,s2:str):
    m = len(s1)
    n = len(s2)
    lcs = longestCommonSubsequence(s1,m,s2,n)

    return m - lcs + n - lcs


"""动态规划思路
# 自顶向下递归的动态规划
def dp(状态1, 状态2, ...):
    for 选择 in 所有可能的选择:
        # 此时的状态已经因为做了选择而改变
        result = 求最值(result, dp(状态1, 状态2, ...))
    return result

# 自底向上迭代的动态规划
# 初始化 base case
dp[0][0][...] = base case
# 进行状态转移
for 状态1 in 状态1的所有取值：
    for 状态2 in 状态2的所有取值：
        for ...
            dp[状态1][状态2][...] = 求最值(选择1，选择2...)
"""


"""leetcode509 斐波列数列
"""
class Solution:
    def __init__(self) -> None:
        self.memo = []

    def clib(self,n:int):
        self.dp(n)

        return self.memo[n]

    def dp(self,n:int):
        """dp函数解法
        n:状态
        dp(n-1) 和 dp(n-2)是选择


        Args:
            n (int): _description_

        Returns:
            _type_: _description_
        """
        if n < 0:
            return -1
        if n == 0 or n == 1: 
            return n
        
        if self.memo[n] != 0:
            return self.memo[n]
        self.memo[n] = self.dp(n-1) + self.dp(n-1)
        
class Solution:
    def clib(self, n:int):
        if n < 0:
            return -1
        
        dp = [0] * (n+1) # dp数组初始化 
        dp[0] = 0
        dp[1] = 1

        for i in range(3,n+1): # 状态
            dp[i] = dp[i-1] + dp[i-2] # i-1, i-2为选择
        return dp[n]

"""leetcode322 凑零钱 
"""
class Solution:
    def __init__(self,n) -> None:
        self.memo= [0] * (n+1)

    def coin_change(self,coins:List[int], amount:int):
        if amount < 0:
            return -1
        return self.dp(coins, amount)

    def dp(self,coins:List[int], amount:int):
        
        if amount == 0: # base case
            return 0
        if amount < 0:
            return -1
        res = float("inf")
        for coin in  coins: # 选择
            sub_res = self.dp(coins,amount-coin)
            if sub_res == -1 :
                res = min(res, sub_res+1)
        return res if res != float("inf") else -1

class Solution:
    def coin_change(self,coins:List, amounts:int):
        if amounts < 0:
            return -1 
        if amounts == 0:
            return 0
        # dp 数组的定义：当目标金额为 i 时，至少需要 dp[i] 枚硬币凑出。
        dp = [float("inf")] * (amounts+1) 
        dp[0] = 0
        for i in range(1,amounts+1): # 状态
            for coin in coins: # 选择
                if i - coin < 0:
                    continue
                dp[i] = min(dp[i],dp[i-coin]+1)
        
        return dp[amounts] if dp[amounts] != float("inf") else -1


"""leetcode300最长递增字序列
输入：nums=[10,9,2,5,3,7,101,18]，
输出：4
解释：其中最长的递增子序列是 [2,3,7,101]，所以算法的输出应该是 4
"""
def length_of_lis(nums:List[int]):
    dp = [0] * len(nums)
    dp[0] = 1
    for i in range(1,len(nums)): 
        for j in range(i): 
            if nums[i] > nums[j]:
                dp[i] = max(dp[i],dp[j]+1) 
    res = 0
    for i in range(len(dp)):
        res = max(res,dp[i])

    return res

class Solution:
    def __init__(self) -> None:
        self.memo = []
    def max_length_of_lis(self,nums:List[int]) -> int :
        self.memo = [-1] * len(nums)
        return self.dp(nums,len(nums)-1)
    
    def dp(self,nums:List[int],i:int):
        """
        leetcode300最长递增字序列
        输入：nums=[10,9,2,5,3,7,101,18]，
        输出：4
        解释：其中最长的递增子序列是 [2,3,7,101]，所以算法的输出应该是 4
        
        dp[i] 表示以 nums[i] 这个数结尾的最长递增子序列的长度。 

        Args:
            nums (List[int]): _description_
        """
        if i == 0:
            return 1
        if self.memo[i] != -1:
            return self.memo[i]
        res = 1
        for j in range(i,-1,-1):
            if nums[i] > nums[j]:
                res = max(res,1+self.dp(nums,j)) # TODO
        self.memo[i] = res
        return res

"""给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），
每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？

输入：8
输出：18
解释：2 * 3 * 3 = 18
"""
class Solution:
    def cutting_rope(self,n:int):
        """解题思路：动态规划
        dp[i] 表示长度为i的绳子剪成m段后的最大乘积
        
        此时我们考虑第二段的情况时，第一段绳子的长度为 2，而事实上，第一段绳子可取的范围为 [ 2，i）。
        此时，假设当下绳子的长度为 i。
        长度为 n 的绳子剪掉后的最大乘积与求绳子长度为 n - 1 、 n - 2 、 n - 3 的最大乘积求法一样。
        假设剪的绳子那段称为 第一段，剪剩下的那段绳子称为 第二段，那么第一段的范围为 [2,i)，第二段可以剪或者不剪，
        假设 dp[i] 表示长度为 i 的绳子剪成 m 段后的最大乘积，
        那么，不剪总长度乘积为 j * （ i - j），剪的话长度乘积为 j * dp[ i - j ]，
        取两者的最大值，即 Max ( j * ( i - j) , j * dp[ i - j] )。

        状态转移方程：dp[i] = Max(dp[i], Max(j * (i - j), j * dp[i - j]))

        Args:
            n (int): _description_

        Returns:
            _type_: _description_
        """
        if n <= 1:
            return 1
        
        dp = [0] * (n+1) # dp数组初始化
        dp[2] = 1 # 绳子必须剪断
        for i in range(3,n+1):
            # 对于长度为 i 的绳子，它可以分为两个区间 j 和 i - j
            # j 的范围由 2 开始，因为剪长度为 1 的绳子无法扩大乘积的值
            # j 的范围可以延伸至 i - 1
            for j in range(2,i):
                dp[i] = max(dp[i],max(j*(i-j),j*dp[i-j]))
        return dp[n]

"""leetcode337 打家劫舍-III
在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。
这个地区只有一个入口，我们称之为根。
除了“根”之外，每栋房子有且只有一个“父“房子与之相连。
一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。
如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。
"""
class Solution:
    def rob(self,root:TreeNode):
        def dp(root):
            # 递归终止条件
            # 即在叶子节点的时候，叶子节点的左右子节点为 null
            # 继续调用下去就会返回给叶子节点，它的左右子树的选择的金额都是 0
            if not root:
                return [0,0]
            # 否则的话，说明当前节点有值
            # 需要判断，当前节点是偷还是不偷
            
            # 1、如果选择偷，那么左子节点不能偷，这个时候需要获取不偷左子节点时的最大金额
            # 2、如果选择偷，那么右子节点不能偷，这个时候需要获取不偷右子节点时的最大金额
            # 3、如果选择不偷，那么可以偷左右子节点，这个时候的最大金额为左右子节点最大金额之和

            # dp[0] 表示的是以当前 node 为根节点的子树能够偷取的最大金额，并且此时采取【不偷】 node 这个节点的策略
            # dp[1] 表示的是以当前 node 为根节点的子树能够偷取的最大金额，并且此时采取【偷】 node 这个节点的策略
            left = dp(root.left)
            right = dp(root.right)

            # 抢，下家就不能抢了
            rob = root.val + left[0] + right[0]
            # 不抢，下家可抢可不抢，取决于收益大小
            not_rob = max(left[0],left[1]) + max(right[0],right[1])
            return [not_rob,rob]
        res = dp(root)
        return max(res[0],res[1])


if __name__ == "__main__":
    nums  = [1,2,3,4,5,6,7]
    print(max_sub_array(nums=[-3,1,3,-1,2,-4,2]))