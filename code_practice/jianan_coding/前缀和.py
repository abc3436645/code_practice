"""前缀和
s[0]=0
s[i+1]=s[i]+a[i]

通过前缀和，我们可以把连续子数组的元素和转换成两个前缀和的差

前缀和数组是严格递增的，因此可以和双指针进行结合，二分查找

如果 left=0，要计算的子数组是一个前缀（从 a[0] 到 a[right]），我们要用 s[right+1]减去 s[0]。
如果不定义 s[0]=0，就必须特判 left=0 的情况了（读者可以试试）。通过定义 s[0]=0，任意子数组（包括前缀）都可以表示为两个前缀和的差。
此外，如果 a是空数组，定义 s[0]=0 的写法是可以兼容这种情况的。
"""
from typing import List

class NumArray:
    def __init__(self,nums:List[int]) -> None:
        """前缀和
        Args:
            nums (List[int]): _description_
        """
        prefix_sum = [0] * (len(nums) + 1) # 前缀和数组初始化
        for i,x in enumerate(nums):
            prefix_sum[i+1] = prefix_sum[i] + x 

        self.prefix_sum = prefix_sum

    def sumRange(self,left:int,right:int):
        res =  self.prefix_sum[right + 1] - self.prefix_sum[left]

        return res 
    

# leetcode 2559. 统计范围内的元音字符串数
"""
输入：words = ["aba","bcb","ece","aa","e"], queries = [[0,2],[1,4],[1,1]]
输出：[2,3,0]
解释：以元音开头和结尾的字符串是 "aba"、"ece"、"aa" 和 "e" 。
查询 [0,2] 结果为 2（字符串 "aba" 和 "ece"）。
查询 [1,4] 结果为 3（字符串 "ece"、"aa"、"e"）。
查询 [1,1] 结果为 0 。
返回结果 [2,3,0] 。
"""
class Solution:
    def vowelStrings(self, words: List[str], queries: List[List[int]]) -> List[int]:
        prefix_sum = [0] * (len(words)+1) # 初始化前缀和数组
        for i, w in enumerate(words):
            prefix_sum[i+1] = prefix_sum[i] 
            if w[0] in "aeiou" and w[-1] in "aeiou": # 判断条件
                prefix_sum[i+1] += 1 
        ans = [0] * len(queries)
        for i,q in enumerate(queries):
            ans[i] = prefix_sum[q[1]+1] - prefix_sum[q[0]]

        return ans 

# leetcode 2389. 和有限的最长子序列
"""
给你一个长度为 n 的整数数组 nums ，和一个长度为 m 的整数数组 queries 。
返回一个长度为 m 的数组 answer ，其中 answer[i] 是 nums 中 元素之和小于等于 queries[i]的子序列的最大长度。
子序列 是由一个数组删除某些元素（也可以不删除）但不改变剩余元素顺序得到的一个数组。


输入：nums = [4,5,2,1], queries = [3,10,21]
输出：[2,3,4]
解释：queries 对应的 answer 如下：
- 子序列 [2,1] 的和小于或等于 3 。可以证明满足题目要求的子序列的最大长度是 2 ，所以 answer[0] = 2 。
- 子序列 [4,5,1] 的和小于或等于 10 。可以证明满足题目要求的子序列的最大长度是 3 ，所以 answer[1] = 3 。
- 子序列 [4,5,2,1] 的和小于或等于 21 。可以证明满足题目要求的子序列的最大长度是 4 ，所以 answer[2] = 4 
"""
class Solution:
    def answerQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        """贪心：由于元素和有上限，为了能让子序列尽量长，子序列中的元素值越小越好。
           对于本题来说，元素在数组中的位置是无关紧要的（因为我们计算的是元素和），所以可以排序了。
           把 nums从小到大排序后，再从小到大选择尽量多的元素（相当于选择一个前缀），使这些元素的和不超过询问值。

           由于 nums[i]都是正整数，前缀和是严格单调递增的，这样就能在前缀和上使用二分查找：
           找到大于 queries[i]的第一个数的下标，由于下标是从0开始的，
           这个数的下标正好就是前缀和小于等于 queries[i]的最长前缀的长度。

        Args:
            nums (List[int]): _description_
            queries (List[int]): _description_

        Returns:
            List[int]: _description_
        """
        nums.sort()
        prefix_nums = [0]*(len(nums)+1)
        for i,n in enumerate(nums):
            prefix_nums[i+1] = prefix_nums[i]+n
        # print(prefix_nums)
        ans = [0] * (len(queries))
        for i,q in enumerate(queries):
            ans[i] = self.upper_bound(prefix_nums,q)
        return ans

    def upper_bound2(self,nums:List[int],target:int):
        left = -1
        right = len(nums) #左开右开写法
        while left+1  < right:
            mid = left + (right-left) // 2
            if nums[mid] > target:
                right = mid
            else:
                left = mid
        return right-1  

    def upper_bound(self,nums:List[int],target:int): 
        """二分查找，找到大于target的第一个数下标

        Args:
            nums (List[int]): _description_
            target (int): _description_

        Returns:
            _type_: _description_
        """
        left = 0
        right = len(nums)-1 # 闭区间写法
        while left <= right:
            mid = left + (right-left) // 2
            if nums[mid] > target:
                right = mid-1
            else:
                left = mid+1
        return right   
    

# leetcode 2438 二的幂数组中查询范围内的乘积
"""
输入：n = 15, queries = [[0,1],[2,2],[0,3]]
输出：[2,4,64]
解释：
对于 n = 15 ，得到 powers = [1,2,4,8] 。没法得到元素数目更少的数组。
第 1 个查询的答案：powers[0] * powers[1] = 1 * 2 = 2 。
第 2 个查询的答案：powers[2] = 4 。
第 3 个查询的答案：powers[0] * powers[1] * powers[2] * powers[3] = 1 * 2 * 4 * 8 = 64 。
每个答案对 109 + 7 得到的结果都相同，所以返回 [2,4,64] 。
"""
class Solution:
    def productQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        
        MOD = 1_000_000_007
        
        powers = [] # 找到最少 数目的 2 的幂，且它们的和为 n 
        while n:
            power = pow(2, int(log2(n)))
            powers.append(power)
            n -= power
        powers = powers[::-1]  # 反转结果变成有序数组
        
        prefix = [0 for _ in range(len(powers))] # 前缀和数组初始化
        prefix[0] = powers[0]
        for i in range(1, len(powers)):    
            prefix[i] = prefix[i - 1] * powers[i]  # 计算前缀数组乘积
  
        res = [0 for _ in range(len(queries))]
        for i, (left, right) in enumerate(queries):
            res[i] = ((prefix[right] // prefix[left]) * powers[left]) % MOD
        
        return res
        
# leetcode 2055. 蜡烛之间的盘子
"""
输入：s = "**|**|***|", queries = [[2,5],[5,9]]
输出：[2,3]
解释：
- queries[0] 有两个盘子在蜡烛之间。
- queries[1] 有三个盘子在蜡烛之间。
"""
class Solution:
    def platesBetweenCandles(self, s: str, queries: List[List[int]]) -> List[int]:
        n = len(s)
        sum_ = [0] * (n + 1)  # sum[i] 表示 s[:i] 中盘子的个数
        left = [0] * n  # left[i] 表示 i 左侧最近蜡烛位置
        p = -1
        for i, b in enumerate(s):
            sum_[i + 1] = sum_[i]
            if b == '|':
                p = i 
            else:
                sum_[i + 1] += 1
            left[i] = p

        right = [0] * n  # right[i] 表示 i 右侧最近蜡烛位置
        for i in range(n - 1, -1, -1):
            if s[i] == '|':
                p = i
            right[i] = p

        ans = []
        for q in queries:
            l, r = right[q[0]], left[q[1]]  # 用最近蜡烛位置来代替查询的范围，从而去掉不符合要求的盘子
            if l < r:
                ans.append(sum_[r] - sum_[l])
            else:
                ans.append(0)
        return ans

# leetcode 53. 最大子数组和
"""_summary_
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。

输入：nums = [5,4,-1,7,8]
输出：23
"""
class Solution:
    def maxSubArray(self, nums: List[int]):
        """
        由于子数组的元素和等于两个前缀和的差，所以求出 nums的前缀和，问题就变成 121. 买卖股票的最佳时机了。本题子数组不能为空，相当于一定要交易一次。
        我们可以一边遍历数组计算前缀和，一边维护前缀和的最小值（相当于股票最低价格），用当前的前缀和（卖出价格）减去前缀和的最小值（买入价格），就得到了以当前元素结尾的子数组和的最大值（利润），用它来更新答案的最大值（最大利润）。
        注意，由于题目要求子数组不能为空，应当先计算前缀和-最小前缀和，再更新最小前缀和。相当于不能在同一天买入股票又卖出股票。
        如果先更新最小前缀和，再计算前缀和-最小前缀和，就会把空数组的元素和0算入答案。

        Args:
            nums (List[int]): _description_

        Returns:
            _type_: _description_
        """
        ans = float("-inf") 
        pre_sum = min_pre_sum = 0
        for x in nums:
            pre_sum += x # 前缀和
            ans = max(ans,pre_sum-min_pre_sum)
            min_pre_sum = min(min_pre_sum,pre_sum)
        return ans 
    
    def maxSubArray(self,nums:List[int]):
        "递归解法"
        n = len(nums)
        self.ans = []
        @cache
        def dfs(i):
            "以nums[i]结尾的最大子数组和为dfs(i)"
            if i == 0:
                self.ans.append(nums[0])
                return nums[i]
            res = max(dfs(i-1)+nums[i],nums[i]) # 递归
            self.ans.append(res)
            return res
        dfs(n-1)
        return max(self.ans)
    
    def maxSubArray3(self,nums:list[int]):
        """动态规划递推解法
        "dp[i]: 以nums[i]结尾的最大子数组和为df[i]"
        Args:
            nums (list[int]): _description_

        Returns:
            _type_: _description_
        """
        n = len(nums)
        if n == 0:
            return 0
        dp = [0] * n
        # base case
        # 第一个元素前面没有子数组
        dp[0] = nums[0]
        # 状态转移方程
        for i in range(1, n):
            dp[i] = max(nums[i], nums[i] + dp[i - 1])
        # 得到 nums 的最大子数组
        res = float('-inf')
        for i in range(n):
            res = max(res, dp[i])
        return res
    
# leetcode 724. 寻找数组的中心索引
"""
给定一个整数类型的数组 nums，请编写一个能够返回数组 “中心索引” 的方法。
我们是这样定义数组 中心索引 的：数组中心索引的左侧所有元素相加的和等于右侧所有元素相加的和。
如果数组不存在中心索引，那么我们应该返回 -1。如果数组有多个中心索引，那么我们应该返回最靠近左边的那一个。
输入：
nums = [1, 7, 3, 6, 5, 6]
输出：3
"""
class Solution:
    def pivotIndex(self,nums:List[int]):
        """
        数组中心索引的左侧所有元素相加的和等于右侧所有元素相加的和
        也就是 left + num[i] + right = sum(nums)
        left == right
        left == (sum(nums) - nums[i]) //2 时 i的索引
        """
        res = []
        nums_sum = sum(nums)
        for i,n in enumerate(nums):
            if sum(nums[:i]) == (nums_sum - nums[i]) / 2 :
                return i 
        return -1 
    
class Solution:
    def pivotIndex(self,nums:List[int]):
        """前缀和解法

        Args:
            nums (List[int]): _description_
        """
        n = len(nums)
        left,right = 0,sum(nums) 
        for i in range(n):
            right -= nums[i]
            if left == right:
                return i
            left += nums[i]

        return -1 

# leetcode 560. 和为 K 的子数组
"""
给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 。
子数组是数组中元素的连续非空序列。

输入：nums = [1,1,1], k = 2
输出：2
"""
from collections import defaultdict

class Solution:
    def subarraySum(self, nums:List[int], k:int):
        n = len(nums)
        d = defaultdict(int)
        d[0] = 1 
        sum_ = 0 
        res = 0
        for i in range(n):
            sum_ += nums[i] 
            if sum_ - k in d:
                res += d[sum_-k]
            d[sum_] += 1 

        return res 
    
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        s = [0] * (len(nums) + 1)
        for i, x in enumerate(nums):
            s[i + 1] = s[i] + x

        ans = 0
        cnt = defaultdict(int)
        for sj in s:
            ans += cnt[sj - k]
            cnt[sj] += 1
        return ans
    
# leetcode 1524. 和为奇数的子数组数目
"""
输入：arr = [1,3,5]
输出：4
解释：所有的子数组为 [[1],[1,3],[1,3,5],[3],[3,5],[5]] 。
所有子数组的和为 [1,4,9,3,8,5].
奇数和包括 [1,9,3,5] ，所以答案为 4 。

这道题要求返回和为奇数的子数组数目。为了快速计算任意子数组的和，可以通过维护前缀和的方式。这道题只需要知道每个子数组的和的奇偶性，不需要知道子数组的和的具体值，因此不需要维护每一个前缀和，只需要维护奇数前缀和的数量与偶数前缀和的数量。
分别使用 odd 和 even表示奇数前缀和的数量与偶数前缀和的数量。初始时，odd=0 even=1，因为空的前缀的和是 0，也是偶数前缀和。
遍历数组 arr并计算前缀和。对于下标 iii 的位置的前缀和（即 arr[0]+arr[1]+…+arr[i]，根据奇偶性进行如下操作：
当下标 iii 的位置的前缀和是偶数时，如果下标 jjj 满足 j<i且下标 j 的位置的前缀和是奇数，则从下标 j+1到下标 i 的子数组的和是奇数，因此，以下标 i 结尾的子数组中，和为奇数的子数组的数量即为奇数前缀和的数量 odd
当下标 i 的位置的前缀和是奇数时，如果下标 j 满足 j<i 且下标 j 的位置的前缀和是偶数，则从下标 j+1到下标 i 的子数组的和是奇数，因此，以下标 iii 结尾的子数组中，和为奇数的子数组的数量即为偶数前缀和的数量 even
"""

class Solution:
    def numOfSubarrays(self, arr: List[int]) -> int:
        MODULO = 10**9 + 7
        odd = 0 # 奇数
        even = 1 # 偶数
        pre_sum = 0
        subarrays = 0 
        for x in arr:
            pre_sum += x
            subarrays += (odd if pre_sum % 2 == 0 else even)
            if pre_sum % 2 == 0:
                even += 1 
            else:
                odd += 1
        return subarrays % MODULO

# leetcode 974 和可以被k整除的子数组
"""
输入：nums = [4,5,0,-2,-3,1], k = 5
输出：7
解释：
有 7 个子数组满足其元素之和可被 k = 5 整除：
[4, 5, 0, -2, -3, 1], [5], [5, 0], [5, 0, -2, -3], [0], [0, -2, -3], [-2, -3]
"""
class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        record = {0: 1} # 初始化，record表示key为前缀和mod k的值，value为个数
        total, ans = 0, 0
        for elem in nums:
            total += elem
            modulus = total % k
            same = record.get(modulus, 0)
            ans += same
            record[modulus] = same + 1
        return ans
    
# leetcode 523 
""" 
子数组大小 至少为 2 ，且
子数组元素总和为 k 的倍数。
如果存在，返回 true ；否则，返回 false 

输入：nums = [23,2,4,6,7], k = 6
输出：true
解释：[2,4] 是一个大小为 2 的子数组，并且和为 6 。
"""
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        """前缀和+字典
        同余定理：即当两个数除以某个数的余数相等，那么二者相减后肯定可以被该数整除
        Args:
            nums (List[int]): _description_
            k (int): _description_

        Returns:
            bool: _description_
        """
        pre_sum = 0
        d = {0:-1} # 余数:下标
        for index,x in enumerate(nums):
            pre_sum = (pre_sum+x)%k
            if pre_sum in d:
                if index - d[pre_sum] >= 2:
                    return True
                else:
                    d[pre_sum] = index
        return False 
