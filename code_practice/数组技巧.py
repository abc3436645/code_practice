"""
在处理数组和链表相关问题时，双指针技巧是经常用到的，双指针技巧主要分为两类：左右指针和快慢指针。

所谓左右指针，就是两个指针相向而行或者相背而行；而所谓快慢指针，就是两个指针同向而行，一快一慢。

对于单链表来说，大部分技巧都属于快慢指针，前文 单链表的六大解题套路 都涵盖了，比如链表环判断，倒数第 K 个链表节点等问题，它们都是通过一个 fast 快指针和一个 slow 慢指针配合完成任务。

在数组中并没有真正意义上的指针，但我们可以把索引当做数组中的指针，这样也可以在数组中施展双指针技巧
"""


# 快慢指针：数组问题中比较常见的快慢指针技巧，是让你原地修改数组。
# LeetCode26 原地修改数组，删除数组里面的重复项
"""
输入：nums = [0,0,1,1,1,2,2,3,3,4]
输出：5, nums = [0,1,2,3,4]
解释：函数应该返回新的长度 5 ， 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4 。不需要考虑数组中超出新长度后面的元素 

输入：nums = [1,1,2]
输出：2, nums = [1,2,_]
解释：函数应该返回新的长度 2 ，并且原数组 nums 的前两个元素被修改为 1, 2 。不需要考虑数组中超出新长度后面的元素。
"""
from typing import List

class Solution:
    def removeDuplicates(self,nums:List[int]):
        """思路：快慢指针，slow指针在后，fast指针在前探路，找到一个不重复的元素赋值给slow,让slow指针向前一步，维护nums[0:slow]
            画图画图画图
        Args:
            nums (List[int]): _description_

        Returns:
            _type_: _description_
        """
        
        if len(nums) == 0:  # base case
            return 0
        slow, fast = 0,0 # 快慢指针
        while fast <= len(nums)-1: # 循环控制条件
            if nums[slow] != nums[fast]:
                slow += 1
                nums[slow] = nums[fast]
            fast += 1
        return slow + 1 # 返回数组长度，数组长度为索引长度+1
    

# LeetCode83 删除排序链表中重复元素
class ListNode:
    def __init__(self,value,next=None):
        self.value = value
        self.next = next

class Solution:
    def removeDuplicates(self,head:ListNode):
        if not head:
            return None
        slow,fast = head,head
        while fast:
            if slow.value != fast.value:
                slow.next = fast # slow指向fast位置
                slow = slow.next # slow指针向前一步
            fast = fast.next # fast指针向前一步
        slow.next = None #断开和后面重复元素的链接
        return head
    

# LeetCode27 移除指定元素
"""
输入：nums = [3,2,2,3], val = 3
输出：2, nums = [2,2]
解释：函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。你不需要考虑数组中超出新长度后面的元素。例如，函数返回的新长度为 2 ，而 nums = [2,2,3,3] 或 nums = [2,2,0,0]，也会被视作正确答案。

输入：nums = [0,1,2,2,3,0,4,2], val = 2
输出：5, nums = [0,1,4,0,3]
解释：函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。注意这五个元素可为任意顺序。你不需要考虑数组中超出新长度后面的元素。
"""

class Solution:
    def removeElement(self,nums:List[int],val:int):
        """思路：快慢指针，画图画图画图
        注意这里和有序数组去重的解法有一个细节差异，我们这里是先给 nums[slow] 赋值然后再给 slow++，
        这样可以保证 nums[0..slow-1] 是不包含值为 val 的元素的，最后的结果数组长度就是 slow。

        Args:
            nums (List[int]): _description_
            val (int): _description_

        Returns:
            _type_: _description_
        """
        if not nums: # base case
            return 0
        slow , fast = 0, 0
        while fast < len(nums):
            if nums[fast] != val:
                nums[slow] =  nums[fast]
                slow += 1
            fast += 1

        return slow


"""leetcode16 最接近的三数之和
"""
def threeSumClosest(nums:List[int],target:int) -> int:
    if len(nums) < 3:
        return 0
    nums.sort()
    closest = float('inf')
    for i in range(len(nums)):
        sum_ = nums[i] + twoSumClosest(nums,i+1,target-nums[i])
        if abs(sum_ - target) < abs(closest - target):
            closest = target - sum_
    return target - closest

def twoSumClosest(nums:List[int],start,target:int) -> int:
    """两数之和最接近

    Args:
        nums (List[int]): _description_
        start (_type_): _description_
        target (int): _description_

    Returns:
        int: _description_
    """
    left,right = start,len(nums)-1
    closest = float('inf')
    while left < right:
        sum = nums[left] + nums[right]
        if abs(sum-target) < abs(closest): # 更新最接近值
            closest = target- sum
        if sum < target:
            left += 1
        else:
            right -= 1
    return target - closest

"""LeetCode244 最短单词距离II
"""
class WordDistance:
    def __init__(self,words:List[str]):
        self.word_map = {}
        for i,word in enumerate(words):
            if word not in self.word_map:
                self.word_map[word] = []
            self.word_map[word].append(i)
    
    def shortest(self,word1:str,word2:str) -> int:
        idx1 = self.word_map[word1]
        idx2 = self.word_map[word2]
        i,j = 0,0
        min_distance = float('inf')
        while i < len(idx1) and j < len(idx2):
            min_distance = min(min_distance,abs(idx1[i]-idx2[j]))
            if idx1[i] < idx2[j]:
                i += 1
            else:
                j += 1
        return min_distance
    
"""leetcode 两数相加"""
def addTwoNumbers(l1:ListNode,l2:ListNode) -> ListNode:
    dummy = ListNode(0)
    cur = dummy
    carry = 0
    while l1 or l2:
        x = l1.value if l1 else 0
        y = l2.value if l2 else 0
        sum = x + y + carry
        carry = sum // 10
        cur.next = ListNode(sum % 10)
        cur = cur.next
        if l1:
            l1 = l1.next
        if l2:
            l2 = l2.next
    if carry:
        cur.next = ListNode(carry)
    return dummy.next