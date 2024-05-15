"""动态规划
1.动态规划问题的一般形式就是求最值.
2.求解动态规划的核心问题是穷举

1. 列举正确的「状态转移方程」
2. 需要判断算法问题是否具备「最优子结构」，是否能够通过子问题的最值得到原问题的最值
3. 动态规划问题存在「重叠子问题」，如果暴力穷举的话效率会很低，所以需要你使用「备忘录」或者「DP table」来优化穷举过程，避免不必要的计算。

动态规划思维框架：明确 base case -> 明确「状态」-> 明确「选择」 -> 定义 dp 数组/函数的含义。


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


动态规划五要素：
    1.确定dp数组（dp table）以及下标的含义
    2.确定递推公式
    3.dp数组如何初始化
    4.确定遍历顺序
    5.举例推导dp数组

"""

# leetcode 509 斐波那契数列
from functools import cache
from typing import List, cast

from code_practice import dynamic_programming


class Solution:
    def fib(self, n: int) -> int:
        self.memo = [0] * (n+1) # 备忘录
        return self.dp(n)


    def dp(self, n: int) -> int:
        """
        
        """
        if n < 0:
            return -1 
        # base case 
        if n == 0 or n == 1 :
            return n
        if self.memo[n]!= 0:
            return self.memo[n]
       self.memo[n] = self.dp(n-1) + self.dp(n-2)

       return self.memo[n]


class Solution:
    def fib(self, n: int) -> int:
        dp = [0] * (n+1)  # dp数组
        # base case 
        dp[0] = 0 
        dp[1] = 1 
        for i in range(2,n+1): # 遍历顺序 状态
            dp[i] = dp[i-1] + dp[i-2] # 状态转移方程

        return dp[n]
        

# leetcode 322 零钱兑换
# 给你 k 种面值的硬币，面值分别为 c1, c2 ... ck，每种硬币的数量无限，再给一个总金额 amount，问你最少需要几枚硬币凑出这个金额，如果不可能凑出，算法返回 -1
#
# 1、确定 base case，这个很简单，显然目标金额 amount 为 0 时算法返回 0，因为不需要任何硬币就已经凑出目标金额了。
# 2、确定「状态」，也就是原问题和子问题中会变化的变量。由于硬币数量无限，硬币的面额也是题目给定的，只有目标金额会不断地向 base case 靠近，所以唯一的「状态」就是目标金额 amount。
# 3、确定「选择」，也就是导致「状态」产生变化的行为。目标金额为什么变化呢，因为你在选择硬币，你每选择一枚硬币，就相当于减少了目标金额。所以说所有硬币的面值，就是你的「选择」。
# 4、明确 dp 函数/数组的定义。我们这里讲的是自顶向下的解法，所以会有一个递归的 dp 函数，一般来说函数的参数就是状态转移中会变化的量，也就是上面说到的「状态」；函数的返回值就是题目要求我们计算的量。就本题来说，状态只有一个，即「目标金额」，题目要求我们计算凑出目标金额所需的最少硬币数量。
# 所以我们可以这样定义 dp 函数：dp(n) 表示，输入一个目标金额 n，返回凑出目标金额 n 所需的最少硬币数量。
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        self.memo = [-666] * (amount + 1)
        return self.dp(coins,amount)

    def dp(self, coins: List[int], amount:int) -> int:
        """
        对应二叉树分解问题思路：定义：要凑出金额 amount，至少要 dp(coins, n) 个硬币
        """
        res = float("inf")
        if amount < 0:
            return -1
        if amount == 0 :
            return 0
        if self.memo[amount] != -666:
            return self.memo[amount]
        for coin in coins:  # N叉树 递归遍历
            sub_problem = self.dp(coins,amount-coin)  # N叉树遍历
            if sub_problem == -1: # 后续遍历
                continue
            res = min(res,sub_problem+1) 
        self.memo[amount] = -1 if res == float("inf") else res
        return self.memo[amount]


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # dp 数组的定义：当目标金额为 i 时，至少需要 dp[i] 枚硬币凑出。
        dp = [amount+1] * (amount+1)
        dp[0] = 0
        # 外层 for 循环在遍历所有状态的所有取值
        for i in range(amount+1):
            # 内层 for 循环在求所有选择的最小值
            for coin in coins:  
                if i -coin < 0: # 子问题无解
                    continue
                dp[i] = min(dp[i],dp[i-coin]+1)
        return -1 if dp[amount] == amount+1 else dp[amount]


# leetcode 300. 最长递增子序列
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # 定义：dp[i] 表示以 nums[i] 这个数结尾的最长递增子序列的长度
        dp = [1] * len(nums)
        dp[1] = 1 
        for i in range(2,len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i],dp[j]+1)
        return max(dp)

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # TODO
        pass

    def dp(self, nums: List[int],i:int) -> int:
        """
        dp函数定义：以nums[i] 这个数结尾的最长递增子序列的长度
        """
        for i in range(len(nums)):
            sub_problem = self.dp()


# leetcode 编辑距离
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        self.memo = []
        m = len(word1)
        n = len(word2)

        return self.dp(word1, m-1, word2, n-1)


    def dp(self, word1: str,i:int, word2: str,j:int) -> int:
        """
        dp函数定义：定义：s1[0..i] 和 s2[0..j] 的最小编辑距离是 dp(s1, i, s2, j)
        """
        # base case
        if i == -1 : # 空串s1
            return j + 1 
        if j == -1 : # 空串s2
            return i+1 
        if word1[i] == word2[j]:
            return self.dp(word1,i-1,word2,j-1)
        else:
            return min(
                self.dp(word1,i-1,word2,j-1) + 1,
                self.dp(word1, i, word2, j-1) + 1,
                self.dp(word1, i-1, word2, j) + 1 
            )

class Solution:
    def minDistance(self,s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        # 定义：s1[0..i] 和 s2[0..j] 的最小编辑距离是 dp[i+1][j+1]
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        # 初始化 base case 
        for i in range(1, m + 1):
           dp[i][0] = i
        for j in range(1, n + 1):
           dp[0][j] = j
    
        # 自底向上求解
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # 进行状态转移
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                         dp[i - 1][j] + 1,
                         dp[i][j - 1] + 1,
                         dp[i - 1][j - 1] + 1
                    )
        # 按照 dp 数组的定义，存储 s1 和 s2 的最小编辑距离
        return dp[m][n]


"""
状态定义？状态转移方程？
思路：选或不选/ 选哪个

思考dfs func要怎么写：入参和返回值，递归到哪里，递归边界和入口
改成记忆化搜索，也就是优化时间和空间复杂度
1:1 翻译成dp数组 递推方式

1:1从递归转换成递推
    1. dfs -> dp数组
    2. 递归 -> 循环
    3. 递归边界 -> 数组初始值

"""

# leetcode 198 打家劫舍
# dfs(i) = max(dfs(i-1),dfs(i-1)+nums[i])  可观察递归顺序
# 递推：dp[i] = max(dp[i-1],dp[i-2]+nums[i])
# 为防止溢出： dp[i+2] = max(dp[i+1],dp[i]+nums[i])

class Solution:
    def rob(self,nums:List[int]):
        n = len(nums)
        cache = [-1] * n 

        # 使用@cache装饰器优化时间复杂度，原理类似于使用哈希表或者哈希数组
        @cache 
        def dfs(i):
            if i < 0:
                return 0
            if cache[i] != -1:
                return cache[i]
            res = max(dfs(i-1),dfs(i-2)+nums[i])
            cache[i] = res
            return res 

        return dfs(n-1)


class Solution:
    def rob(self,nums:List[int]):
        n = len(nums)
        dp = [0] * (n+2)
        for i,x in enumerate(nums):
            dp[i+2] = max(dp[i+1],dp[i]+x)

        return dp[n+1]

    def rob2(self,nums:List[int]):
        n = len(nums)
        dp = [0] * (n+2)
        for i x in enumerate(nums):
            dp[i+1] = max(dp[i+1],dp[i]+x)

        return dp[n+1]

    def bob3(self,nums:List[int]):
        n = len(nums)
        dp0 = dp1 = 1 
        for i,x in  enumerate(nums):
            new_dp = max(dp1,dp0+x)
            dp0 = dp1 
            dp1 = new_dp

        return dp1 


# leetcode 01背包
# capacity 背包容量
# w[i]: 第i个物品的体积
# v[i]: 第i个物品的价值
# 返回：所选物品体积不超过capacity的前提下，所能得到的最大价值和
def zero_one_knapsack(capacity:int,w:List[int],v:List[int]):
    n = len(w)
    
    @cache
    def dfs(i,c):
        if i < 0:
            return 0 
        if c < w[i]:
            return dfs(i-1,c) 
        return max(dfs(i-1,c),dfs(i-1,c-w[i]) + v[i])

    return dfs(n-1,capacity)

# leetcode 494 目标和
class Solution:
    def findTargetSumWay(self,nums:List[int],target:int):
        # 假设正数的和为p 
        # 负数和为s-p 
        # p - (s-p) = target 
        # p = (s+t) /2 
        # 问题转换为找出正数和为(s+p)/2的方案数了
        target += sum(nums)
        if target < 0 or target % 2 == 1:
            return 0 
        target //= 2 
        n = len(nums)          
    
        @cache
        def dfs(i,c):
            if i < 0:
                return 1 if c == 0 else 0
            if c < nums[i]:
                return dfs(i-1,c) 
            return dfs(i-1,c) + dfs(i-1,c-nums[i])

        return dfs(n-1,target)
    
    def findTargetSumWay2(self,nums:List[int],target:int):
        target += sum(nums)
        if target < 0 or target % 2 == 1:
            return 0 
        target //= 2 
        n = len(nums)
        dp = [[0] * (target+1) for _ in range(n+1)]
        dp[0][0] == 1 
        for i, x in enumerate(nums):
            for c in range(target+1):
                if c < x:
                    dp[i+1][c] = f[i][c]
                else:
                    dp[i+1][c] = dp[i][c] + f[i][c-x]

        return dp[n][target]

