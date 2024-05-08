from typing import List


### 275
class Solution:
    def hIndex(self, citations: list) -> int:
        n = len(citations)
        left,right = 0,len(citations)-1
        while left<= right:
            pivot = left + (right-left)//2
            if citations[pivot] == n -pivot:
                return n-pivot
            elif citations[pivot] < n-pivot:
                left = pivot+1
            else:
                right = pivot-1
        return n-left

def twoSum(nums:list,target:int):
    """两数相加"""
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []   

def twoSum2(nums:list,target:int):
    """两数相加"""
    hashmap = {}
    for i in range(len(nums)):
        if hashmap.get(target-nums[i]) is not None:
            return [hashmap.get(target-nums[i]),i]
        hashmap[nums[i]] = i
    return []

def twoSum3(nums:list,target:int):
    """有序数组两数相加"""
    left,right = 0,len(nums)-1
    while left < right:
        if nums[left] + nums[right] == target:
            return [left,right]
        elif nums[left] + nums[right] < target:
            left += 1
        else:
            right -= 1
    return []

"""leetcode 5 最长回文子串
给你一个字符串 s，找到 s 中最长的回文
子串
。

如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。

示例 1：

输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
示例 2：

输入：s = "cbbd"
输出："bb"
"""
class Solution:
    def longestPalindrome(self, s: str) -> str:
        """解题思路：从中心向两边扩展，找到最长的回文子串，分为奇数和偶数两种情况

        Args:
            s (str): _description_

        Returns:
            str: _description_
        """
        if len(s) < 2:
            return s
        max_len = 1
        res = s[0]
        for i in range(len(s)):
            odd,even = self.palindrome(s,i,i),self.palindrome(s,i,i+1)
            maxstr = odd if len(odd) > len(even) else even
            if len(maxstr) > max_len:
                max_len = len(maxstr)
                res = maxstr
        return res
    
    def palindrome(self,s:str,left:int,right:int):
        """
        解题思路：从中心向两边扩展，找到最长的回文子串
        Args:
            s (str): _description_
            left (int): _description_
            right (int): _description_

        Returns:
            _type_: _description_
        """
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left+1:right]

class Solution:
    def longestPalindrome(self, s: str) -> str:
        res = "" 
        for i in range(len(s)):
            s1 = self.palindrome(s,i,i)
            s2 = self.palindrome(s,i,i+1)
            res = res if len(res) > len(s1) else s1
            res = res if len(res) > len(s2) else s2

        return res

    def palindrome(self,s:str,left:int,right:int):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left+1:right]



# 滑动窗口
def slidingWindow(s: str):
    # 用合适的数据结构记录窗口中的数据
    window = {}
    
    left = 0
    right = 0
    
    while right < len(s):
        # c 是将移入窗口的字符
        c = s[right]
        if c not in window:
            window[c] = 1
        else:
            window[c] += 1
            
        # 增大窗口
        right += 1
        
        # 进行窗口内数据的一系列更新
        # ...
        
        # 判断左侧窗口是否要收缩
        while left < right and window needs shrink:
            # d 是将移出窗口的字符
            d = s[left]
            
            # 缩小窗口
            left += 1
            
            # 进行窗口内数据的一系列更新
            # ...

"""leetcode42 接雨水
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 

核心思路：双指针，左右指针分别从两端向中间移动，记录左右两端的最大值，如果左端最大值小于右端最大值，说明左端可以接雨水，反之右端可以接雨水
water[i] = min(
    # 左边最高的柱子
    max(height[0..i]),  
    # 右边最高的柱子
    max(height[i..end]) 
) - height[i]
    
"""
def trap1(helght: list) -> int:
    """暴力解法

    Args:
        helght (list): _description_

    Returns:
        int: _description_
    """
    n = len(helght)
    res = 0
    for i in range(1,n-1):
        left_max = max(helght[:i])
        right_max = max(helght[i+1:])
        res += min(left_max,right_max) - helght[i] if min(left_max,right_max) > helght[i] else 0
    return res

def trap2(height: list) -> int:
    """暴力解法+备忘录

    Args:
        height (list): _description_

    Returns:
        int: _description_
    """
    n = len(height)
    res = 0
    l_max = [] # 数组备忘录
    r_max = [] # 数组备忘录
    # 初始化 base case
    l_max[0] = height[0]
    r_max[n - 1] = height[n - 1]
    for i in range(n):
        l_max[i] = max(height[i],l_max[i-1])
    for i in range(n-1,-1,-1):
        r_max[i] = max(height[i],r_max[i+1])
    for i in range(1,n-1):
        res += min(l_max[i],r_max[i]) - height[i] if min(l_max[i],r_max[i]) > height[i] else 0
    return res


def trap(height: list) -> int:
    """解题思路：双指针，左右指针分别从两端向中间移动，记录左右两端的最大值，如果左端最大值小于右端最大值，说明左端可以接雨水，反之右端可以接雨水

    Args:
        height (list): _description_

    Returns:
        int: _description_
    """
    left,right = 0,len(height)-1
    left_max,right_max = 0,0
    res = 0
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                res += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                res += right_max - height[right]
            right -= 1
    return res

"""leetcode26 删除有序数组中的重复项
输入：nums = [1,1,2]
输出：2, nums = [1,2,_]
解释：函数应该返回新的长度 2 ，并且原数组 nums 的前两个元素被修改为 1, 2 。不需要考虑数组中超出新长度后面的元素。

输入：nums = [0,0,1,1,1,2,2,3,3,4]
输出：5, nums = [0,1,2,3,4]

解题思路：双指针，快指针遍历数组，慢指针记录不重复的元素
"""
def removeDuplicates(nums: list) -> int:
    """双指针

    Args:
        nums (list): _description_

    Returns:
        int: _description_
    """
    if not nums:
        return 0
    slow = 0
    for fast in range(1,len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1

def removeDuplicates2(nums: list) -> int:
    """双指针

    Args:
        nums (list): _description_

    Returns:
        int: _description_
    """
    if not nums:
        return 0
    slow, fast = 0, 1
    while fast < len(nums):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
        fast += 1
    return slow + 1

"""leetcode189 旋转数组
给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。
"""
def rotate(nums: list, k: int) -> None:
    """解题思路：三次翻转
    1. 翻转整个数组
    2. 翻转前k个元素
    3. 翻转后n-k个元素

    Args:
        nums (list): _description_
        k (int): _description_
    """
    n = len(nums)
    k %= n
    def reverse(nums,left,right):
        while left < right:
            nums[left],nums[right] = nums[right],nums[left]
            left += 1
            right -= 1
    reverse(nums,0,n-1)
    reverse(nums,0,k-1)
    reverse(nums,k,n-1)


"""leetcode88 合并两个有序数组
给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。

解题思路：双指针，从后往前遍历，将两个数组中的元素从大到小放入nums1中
"""
def merge(nums1: list, m: int, nums2: list, n: int) -> None:
    """_summary_

    Args:
        nums1 (list): _description_
        m (int): _description_
        nums2 (list): _description_
        n (int): _description_
    """
    i,j = m-1,n-1
    while i >= 0 and j >= 0:
        if nums1[i] > nums2[j]:
            nums1[i+j+1] = nums1[i]
            i -= 1
        else:
            nums1[i+j+1] = nums2[j]
            j -= 1
    nums1[:j+1] = nums2[:j+1] # 如果nums2中还有元素，将其放入nums1中

"""leetcode283 移动零
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

解题思路：双指针，快指针遍历数组，慢指针记录非零元素的位置，遍历结束后将慢指针后的元素全部置为0
"""
def moveZeroes(nums: list) -> None:
    """_summary_

    Args:
        nums (list): _description_
    """
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow += 1
    for i in range(slow,len(nums)):
        nums[i] = 0

"""leetcode66 加一
给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
你可以假设除了整数 0 之外，这个整数不会以零开头。

解题思路：从后往前遍历数组，如果当前元素小于9，直接加1返回，如果当前元素等于9，将当前元素置为0，继续遍历
"""
def plusOne(digits: list) -> list:
    """_summary_

    Args:
        digits (list): _description_

    Returns:
        list: _description_
    """
    n = len(digits)
    for i in range(n-1,-1,-1):
        if digits[i] < 9:
            digits[i] += 1
            return digits
        digits[i] = 0
    return [1] + digits

"""leetcode350 两个数组的交集 II
给定两个数组，编写一个函数来计算它们的交集。

解题思路：哈希表，遍历第一个数组，将元素存入哈希表中，遍历第二个数组，如果元素在哈希表中，将元素存入结果数组中，并将哈希表中的元素个数减1
"""
def intersect(nums1: list, nums2: list) -> list:
    """_summary_

    Args:
        nums1 (list): _description_
        nums2 (list): _description_

    Returns:
        list: _description_
    """
    hashmap = {}
    res = []
    for num in nums1:
        hashmap[num] = hashmap.get(num,0) + 1
    for num in nums2:
        if hashmap.get(num) is not None and hashmap.get(num) > 0:
            res.append(num)
            hashmap[num] -= 1
    return res

def binary_search(nums:List[int], target:int):
    """二分查找，输入有序数组nums和目标targert,如果找到则返回下标，否则返回-1

    Args:
        nums (_type_): _description_
        target (int): _description_
    """
    left, right = 0, len(nums)-1
    while left <= right:
        mid =  left + (right-left) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1 
        elif nums[mid] > target:
            right = mid -1 
    
    return -1 
        
"""leetcode27 移除元素
输入：nums=[1,2,2,3],val=2
输出：nums=[1,3]

输入：nums=[3,2,2,3],val=3
输出：nums=[2,2]
"""
class Solution:
    def remove_element(self,nums:List[int],val:int):
        """暴力解法, 双层循环

        Args:
            nums (List[int]): _description_
            val (int): _description_
        """
        nums_len = len(nums)-1
        for i in range(nums_len):
            if nums[i] == val:
                for j in range(i+1,nums_len):
                    nums[j-1] = nums[j]
            i -= 1
            nums_len -= 1

        return nums_len

    def remove_element_2(self,nums:List[int], val:int):
        slow, fast = 0,0 
        while fast 

if __name__ == "__main__":
    citations = [0,1,3,5,6]
    sol = Solution()
    print(sol.hIndex(citations))