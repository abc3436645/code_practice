# leetcode 283移动0
"""
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

请注意 ，必须在不复制数组的情况下原地对数组进行操作。

示例 1:

输入: nums = [0,1,0,3,12]
输出: [1,3,12,0,0]
示例 2:

输入: nums = [0]
输出: [0]

"""
from typing import List

def moveZeroes(nums: List[int]):
    """
    思路：双指针,快慢指针，交换快慢指针的值，不断维护nums[0:left]
    将nums[left:]所有元素替换为0
    """
    left,right = 0,0
    while right <= len(nums)-1:
        if nums[right] != 0:
            nums[left] = nums[right]
            left += 1
        right += 1 
    for i in range(left,len(nums)): # 替换nums[left:]的值为0
        nums[i] = 0

    print(nums)
    return nums

# leetcode27 移除元素
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        """如果 fast 遇到需要去除的元素，则直接跳过，否则就告诉 slow 指针，并让 slow 前进一步。

        Args:
            nums (List[int]): _description_
            val (int): _description_

        Returns:
            int: _description_
        """
        left,right = 0,0
        while right <= len(nums)-1:
            if nums[right] != val:
                nums[left] = nums[right]
                left += 1
            right += 1 
        print(nums[:left])
        return left
    
# leetcode11 盛最多的水、
"""
用 left 和 right 两个指针从两端向中心收缩，一边收缩一边计算 [left, right] 之间的矩形面积，取最大的面积值即是答案。
不过肯定有读者会问，下面这段 if 语句为什么要移动较低的一边：
// 双指针技巧，移动较低的一边
if (height[left] < height[right]) {
    left++;
} else {
    right--;
}
其实也好理解，因为矩形的高度是由 min(height[left], height5[right]) 即较低的一边决定的：

你如果移动较低的那一边，那条边可能会变高，使得矩形的高度变大，进而就「有可能」使得矩形的面积变大；
相反，如果你去移动较高的那一边，矩形的高度是无论如何都不会变大的，所以不可能使矩形的面积变得更大。

Returns:
    _type_: _description_
"""
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left,right = 0,len(height)-1
        res = 0
        while left < right:
            cur_area = min(height[left],height[right]) * (right-left)
            res = max(res,cur_area)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1 

        return res
    

# leetcode 两数之和,nums元素不重复
class Solution:
    def twoSum(self,nums:List[int],target:int):
        result = []
        left , right = 0, len(nums)-1
        nums.sort()
        while left < right:
            _sum = nums[left] + nums[right]
            if _sum < target :
                left = left + 1
            elif _sum > target:
                right = right - 1
            elif _sum == target:
                result.append([nums[left],nums[right]])
                left = left + 1
                right = right - 1
        return result

# leetcode 两数之和 nums元素存在重复[1,2,2,3,4,4,5,5,6,6,7,8,9]
class Solution:
    def twoSum(self,nums:List[int],target:int) -> List[int]:
        result = [] 
        lo , ro = 0, len(nums)-1
        nums.sort()
        left, right = nums[lo], nums[ro]
        while lo < ro:
            _sum = left + right
            if _sum < target:
                while lo < ro and nums[lo] == left:
                    lo += 1 
            elif _sum > target:
                while lo < ro and nums[ro] == right:
                    ro -= 1
            elif _sum == target:
                result.append([left, right])
                while lo < right and nums[lo] == left:
                    lo += 1
                while ro < right and nums[ro] == right:
                    ro -= 1
        return result
    
def twoSumTarget(nums:List[int],start:int,target:int) -> List[int]:
    result = [] 
    lo , ro = start, len(nums)-1
    nums.sort()
    left, right = nums[lo], nums[ro]
    while lo < ro:
        _sum = left + right
        if _sum < target:
            while lo < ro and nums[lo] == left:
                lo += 1 
        elif _sum > target:
            while lo < ro and nums[ro] == right:
                ro -= 1
        elif _sum == target:
            result.append([left, right])
            while lo < right and nums[lo] == left:
                lo += 1
            while ro < right and nums[ro] == right:
                ro -= 1
    return result

# leetcode 三数之和，nums元素存在重复
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        # n 为 3，从 nums[0] 开始计算和为 0 的三元组
        return self.nSumTarget(nums, 3, 0, 0)

    # 注意：调用这个函数之前一定要先给 nums 排序
    # n 填写想求的是几数之和，start 从哪个索引开始计算（一般填 0），target 填想凑出的目标和
    def nSumTarget(self, nums: List[int], n: int, start: int, target: int) -> List[List[int]]:
        sz = len(nums)
        res = []
        # 至少是 2Sum，且数组大小不应该小于 n
        if n < 2 or sz < n:
            return res
        # 2Sum 是 base case
        if n == 2:
            # 双指针那一套操作
            lo, hi = start, sz - 1
            while lo < hi:
                s = nums[lo] + nums[hi]
                left, right = nums[lo], nums[hi]
                if s < target:
                    while lo < hi and nums[lo] == left:
                        lo += 1
                elif s > target:
                    while lo < hi and nums[hi] == right:
                        hi -= 1
                else:
                    res.append([left, right])
                    while lo < hi and nums[lo] == left:
                        lo += 1
                    while lo < hi and nums[hi] == right:
                        hi -= 1
        else:
            # n > 2 时，递归计算 (n-1)Sum 的结果
            for i in range(start, sz):
                sub = self.nSumTarget(nums, n - 1, i + 1, target - nums[i])
                for arr in sub:
                    # (n-1)Sum 加上 nums[i] 就是 nSum
                    arr.append(nums[i])
                    if arr not in res:
                        res.append(arr)
                while i < sz - 1 and nums[i] == nums[i + 1]:
                    i += 1
        return res


# 合并两个有序链表
"""
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
"""
class ListNode:
    def __init__(self,val,next=None):
        self.val = val
        self.next = next

class Solution:
    def merge_two_list(self,l1:ListNode,l2:ListNode):
        pre_head = ListNode(val=-1)
        result = pre_head
        while l1 and l2:
            if l1.val < l2.val:
                pre_head.next = l1
                l1 = l1.next
            elif l1.val >= l2.val:
                pre_head.next = l2
                l2 = l2.next
            pre_head = pre_head.next

        if l1:
            pre_head.next = l1
        if l2:
            pre_head.next = l2

        return result.next
    
"""
合并K个有序列表

Input:
[
 1->4->5,
 1->3->4,
 2->6
]
Output: 1->1->2->3->4->4->5->6

思路：
借助分治的思想，把 K 个有序链表两两合并即可。相当于是第 21 题的加强版
"""
def merge_two_list(l1:ListNode, l2:ListNode):
    if not l1 :
        return l2
    if not l2 :
        return l1
    pre_head = ListNode(val=-1)
    result = pre_head
    while l1 and l2 :
        if l1.val < l2.val:
            pre_head.next = l1
        else:
            pre_head.next = l2
        pre_head = pre_head.next

    if l1:
        pre_head.next = l1
    if l2:
        pre_head.next = l2

    return result.next

def merge_two_list_2(l1:ListNode, l2:ListNode):
    if not l1:
        return l2
    if not l2:
        return l1
    if l1.val < l2.val:
        merge_two_list_2(l1.next, l2)
        return l1
    else:
        merge_two_list(l1,l2.next)
        return l2

def merge_k_list(k_list:List[ListNode]):
    if not k_list: 
        return 
    if len(k_list) == 1:
        return k_list
    nums = len(nums) // 2 
    left = merge_k_list(k_list[:nums])
    right = merge_k_list(k_list[nums:])

    return merge_two_list(left, right)

# leetcode21 环形链表
"""
给定一个链表，判断链表中是否有环。为了表示给定链表中的环，我们使用整数 pos 来表示链表
尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。

输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。

输入：head = [1], pos = -1
输出：false
解释：链表中没有环。

"""
class Solution:
    def has_cycle(self, head:ListNode):
        # 哈希表判定
        visited = set()
        while head:
            if head in visited:
                return True
            visited.add(head)
            head = head.next
        return False
    
    def has_cycle2(self, head:ListNode):
        if not head or not head.next:
            return False
        slow, fast = head, head.next
        while slow != fast:
            if not fast or not fast.next:
                return False
            slow = slow.next
            fast = fast.next.next

        return True
    
"""leetcode 环形链表入口
给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

输入：head = [3,2,0,-4], pos = 1
输出：tail connects to node index 1

输入：head = [1,2], pos = 0
输出：tail connects to node index 0
"""
def detect_cycle(head:ListNode):
    visited = set()
    while head:
        if head in visited:
            return head
        visited.add(head)
        head = head.next
    return None

def detect_cycle2(head:ListNode):
    if not head or not head.next:
        return None
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    if not fast or not fast.next:
        return None
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return slow
    
# leetcode 两数之和
"""_summary_
在数组中找到 2 个数之和等于给定值的数字，结果返回 2 个数字在数组中的下标。

Given nums = [2, 7, 11, 15], target = 9,
Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1]

解题思路：这道题最优的做法时间复杂度是 O(n)。
顺序扫描数组，对每⼀个元素，在 map 中找能组合给定值的另⼀半数字，如果找到了，直接返回 2 个数字的下标
即可。如果找不到，就把这个数字存⼊ map 中，等待扫到“另⼀半”数字的时候，再取出来返回结果。
"""

def two_sum(nums:List[int],target:int):
    map_dict = {}
    for i in range(len(nums)):
        another = target - nums[i]
        if another in map_dict:
            return [map_dict[another],i]
        elif another not in map_dict:
            map_dict[nums[i]] = i

    return [-1,-1]
    
 
"""逆序链表相加: 2 个逆序的链表，要求从低位开始相加，得出结果也逆序输出，返回值是逆序结果链表的头结点。
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.

解题思路：
需要注意的是各种进位问题：
    Input: (9 -> 9 -> 9 -> 9 -> 9) + (1 -> )
    Output: 0 -> 0 -> 0 -> 0 -> 0 ->1

为了处理⽅法统⼀，可以先建⽴⼀个虚拟头结点，这个虚拟头结点的 Next 指向真正的 head，这样 head 不需要单
独处理，直接 while 循环即可。另外判断循环终⽌的条件不⽤是 p.Next ！= nil，这样最后⼀位还需要额外计算，
循环终⽌条件应该是 p != nil。
"""
def add_two_linklist(l1:List[ListNode],l2:List[ListNode]):
    pre_head = ListNode(val=-1) # 虚拟头结点
    # base case
    if not l1: 
       return l2
    if not l2:
       return l1
    n1, n2, carry, current = 0,0,0,pre_head
    while l1 or l2:
        if not l1 :
            n1 = 0
        else:
            n1 = l1.val
            l1 = l1.next
        if not l2:
            n2 = 0
        else:
            n2 = l2.val
            l2 = l2.next 
        current.next = ListNode(val=(n1+n2+carry)%10)
        current = current.next
        carry = (n1 + n2 + carry) // 10

    return pre_head.next


"""leetcode301 反转字符串
输入：["h","e","l","l","o"]
输出：["o","l","l","e","h"]

输入：["H","a","n","n","a","h"]
输出：["h","a","n","n","a","H"]

思路：使用双指针进行反转字符串。
    假设输入字符串为["h","e","l","l","0"]
    定义left和right分别指向首元素和尾元素
    当left < right ，进行交换。
    交换完毕，left++，right--
    直至left == right

"""
def reverse_string(s:List[str]):
    left, right = 0, len(s)-1
    while left <= right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1

    return s


"""反转数字
Input: 123
Output: 321

Input: -123
Output: -321

Input: 120
Output: 21

思路：这⼀题是简单题，要求反转 10 进制数。类似的题⽬有第 190 题。
这⼀题只需要注意⼀点，反转以后的数字要求在 [−2^31, 2^31 − 1]范围内，超过这个范围的数字都要输出 0
"""
def reverse_number(x:int):
    tmp = 0
    x_abs = abs(x)
    while x_abs != 0:
        tmp = tmp * 10 + x_abs % 10
        x_abs = x_abs // 10
    if x > 0:
        tmp = tmp
    else:
        tmp = -1 * tmp
    if tmp > 2**31 - 1 or tmp < -(2**31):
        return 0
    return tmp


"""判断一个数字是否是回文数字
Input: 121
Output: true

Input: -121
Output: false

Input: 10
Output: false

思路：判断⼀个整数是不是回⽂数。
简单题。注意会有负数的情况，负数，个位数，10 都不是回⽂数。其他的整数再按照回⽂的规则判断
"""
def is_palindrome(x:int):
    if x < 0:
        return False
    elif x  == 0:
        return True
    elif x % 10 == 0:
        return False
    arr = []
    while x > 0 :
        arr.append(x%10)
        x = x // 10
    print("array:", arr)
    # 双指针判断
    left , right = 0, len(arr) - 1 
    while left <= right:
        if arr[left] == arr[right]:
            left = left + 1
            right = right - 1
        else:
            return False
    return True


"""最大雨水
给出⼀个⾮负整数数组 a1，a2，a3，…… an，每个整数标识⼀个竖⽴在坐标轴 x 位置的⼀堵⾼度为 ai 的墙，选择
两堵墙，和 x 轴构成的容器可以容纳最多的⽔

Input: [1,8,6,2,5,4,8,3,7]
Output: 49

思路：
这⼀题也是对撞指针的思路。⾸尾分别 2 个指针，每次移动以后都分别判断⻓宽的乘积是否最⼤
"""
def max_area(nums:List[int]):
    result = 0
    left , right = 0, len(nums) -1 
    while left < right:
        width = abs(right - left)
        if nums[left] < nums[right]:
            height = nums[left]
            left = left + 1
        else:
            height = nums[right]
            right = right - 1

        area = height * width
        if area > result:
            result = area 
    return result


def maxArea(nums):
    max_area = 0
    start, end = 0, len(nums) - 1

    while start < end:
        width = end - start
        high = min(nums[start], nums[end])
        temp = width * high
        if temp > max_area:
            max_area = temp

        if nums[start] < nums[end]:
            start += 1
        else:
            end -= 1

    return max_area


""" 3Sum和
思路：我们要找到满足nums[i] + nums[j] + nums[k] == 0的三元组，那么如果3个数之和等于0，我们可以得出如下两个结论：
【结论1】3个数字的值都是0；
【结论2】3个数字中有正数也有负数；
基于如上分析，我们为了便于进行遍历计算，我们先将nums中的数字进行排序操作。然后我们通过指针i去遍历整个排序后的数组，
与此同时，我们创建两个指针p和q，p指向i+1的位置，q执行数组的末尾。
通过nums[i] + nums[p] + nums[q]我们可以得到总和sum，然后我们进行如下逻辑判断：
【如果sum等于0】则满足题目中的条件，将其放入到List中，后续作为返回值；
【如果sum大于0】我们需要向左移动q指针，因为这样q会变小，整体的sum值也会变小更加趋近于0；
【如果sum小于0】我们需要向右移动p指针，因为这样p会变大，整体的sum值也会变大更加趋近于0；

除了上面的逻辑，我们还需要注意去重的操作，也就是说，当我们移动i指针、p指针或q指针的时候，
如果发现待移动的位置数字与当前数字相同，那么就跳过去继续指向下一个元素，直到找到与当前数字不同的数字为止（当然，要避免数组越界）。
在移动p指针和q指针的过程中，如果不满足p<q，则结束本轮操作即可。
"""
def three_sum(nums:List[int]):
    result = []
    nums.sort()
    print("nums:", nums)
    for i in range(len(nums)-1):
        if i > 0 and nums[i] == nums[i-1]: # 去重
            continue
        j = i + 1 
        k = len(nums) - 1 
        while j < k and nums[k] >= 0:
            iv = nums[i]
            jv = nums[j]
            kv = nums[k]
            theree_nums_sum = iv + jv + kv
            if theree_nums_sum  == 0:
                result.append([iv, jv, kv])
                while j < k and nums[j] == jv: 
                    j += 1
                while j < k and nums[k] == kv:
                        k -= 1
            elif theree_nums_sum > 0 :
                    k -= 1  
            elif theree_nums_sum < 0:
                    j += 1  

    return result

"""3Sum和接近target
Given array nums = [-1, 2, 1, -4], and target = 1.
The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).

解题思路：
⽤两个指针夹逼的⽅法。先对数组进⾏排序，i 从头开始往后⾯扫。这⾥同样需要注意数组中存在
多个重复数字的问题。具体处理⽅法很多，可以⽤ map 计数去重。这⾥笔者简单的处理，i 在循环的时候和前⼀个
数进⾏⽐较，如果相等，i 继续往后移，直到移到下⼀个和前⼀个数字不同的位置。j，k 两个指针开始⼀前⼀后夹
逼。j 为 i 的下⼀个数字，k 为数组最后⼀个数字，由于经过排序，所以 k 的数字最⼤。j 往后移动，k 往前移动，
逐渐夹逼出最接近 target 的值。
这道题还可以⽤暴⼒解法，三层循环找到距离 target 最近的组合。具体⻅代码
"""
def three_sum_closest(nums:List[int],target:int):
    result = 0
    diff = float("inf")
    nums.sort()
    for i in range(len(nums)-1):
        j = i + 1
        k = len(nums) - 1 
        while j < k:
            iv = nums[i]
            jv = nums[j]
            kv = nums[k]
            three_num_sum = iv + jv + kv
            if abs(three_num_sum-target) < diff:
                result = three_num_sum
                diff = abs(three_num_sum-target)
            if three_num_sum == target:
                return result,[iv, jv, kv]
            elif three_num_sum > target:
                k -= 1
            elif three_num_sum < target:
                j += 1 
    return result

"""有效的括号
Input: "()"
Output: true

Input: "()[]{}"
Output: true

Input: "(]"
Output: false

思路：遇到左括号就进栈push，遇到右括号并且栈顶为与之对应的左括号，就把栈顶元素出栈。最后看栈⾥⾯还有没有
其他元素，如果为空，即匹配。
需要注意，空字符串是满⾜括号匹配的，即输出 true。
"""
def vaild_brackets(s:str):
    stack = []
    if not s:
        return True
    for i in range(len(s)):
        if s[i] == '[' or s[i] == '(' or s[i] =='{':
            stack.append(s[i])
        elif s[i] == ']' and len(stack) > 0 and stack[-1] == '[' or \
            s[i] == ')' and len(stack) > 0 and stack[-1] == '(' or \
            s[i] == '}' and len(stack) > 0 and stack[-1] == '{':
                stack = stack[:len(stack)-1]
        else:
            return False

    return len(stack) == 0 

"""leetcode 24 两两交换链表中的节点
【输入】head = [1,2,3,4]
【输出】[2,1,4,3]

【输入】head = [1]
【输出】[1]

思路：3.1> 思路1：遍历交换
根据题目描述，我们需要两两交换节点，然后将最终交换后的链表的头节点返回回来。那么第一个解题思路就是我们通过遍历链表中的节点，然后进行交换操作。
为了方便起见，我们可以在原链表的头节点前面再创建一个虚拟节点Node(-1)，然后创建两个指针p1和p2，p1指向虚拟节点，p2指向原链表的头节点（即：Node(-1)的next节点），
这样，我们就可以通过一下逻辑实现节点交换了，以输入head = [1,2,3,4,5]为例，即：

【步骤1】通过调用ListNode t = p2.next.next暂存Node(3)节点；
【步骤2】通过调用p1.next = p2.next来将Node(-1)链接到Node(2)节点；
【步骤3】通过调用p2.next.next = p2来将Node(2)链接到Node(1)节点；
【步骤4】通过调用p2.next = t来将Node(1)链接到Node(3)节点；
【交换结果】此时链表就变为了Node(-1)——>Node(2)——>Node(1)——>Node(3)——>……了。

执行了一次两个相邻节点交换操作之后，我们需要同时移动p1和p2指针，即：

【移动p2指针】p2 = p2.next;
【移动p1指针】p1 = p1.next.next;

"""
def swap_pair(head:ListNode):
    """迭代遍历

    Args:
        head (ListNode): _description_

    Returns:
        _type_: _description_
    """
    if not head:
        return head 
    pre_head = ListNode(val=-1,next=head) # 虚拟头结点
    p1 = pre_head
    p2 = head
    while p2 and p2.next:
        temp = p2.next.next
        p1.next = p2.next 
        p2.next = temp 
        
        p2 = p2.next
        p1 = p1.next.next

    return pre_head.next

def swap_pair2(head:ListNode):
    if not head or not head.next:
        return head
    new_node = head.next
    head.next = swap_pair2(head.next.next)
    new_node.next = head

    return new_node

def reverse_n_linklist(head:ListNode,n:int):
    """反转链表前 N 个节点

    Args:
        head (ListNode): _description_
        n (int): _description_
    """
    if not head or not head.next:
        return head 
    if n == 1 :
        successor = head.next # 记录后驱节点
        return head 
    last = reverse_n_linklist(head.next,n-1) # 反转n-1节点
    head.next.next = head 
    head.next = successor 

    return last 

"""leetcode20 有效的括号
给定一个只包括 ‘(‘，’)’，'{‘，’}’，'[‘，’]’ 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

1、左括号必须用相同类型的右括号闭合。
2、左括号必须以正确的顺序闭合。

有效的括号满足以下几个条件：

1、字符串的长度一定是偶数。
2、括号的匹配遵循右括号和最近的一个左括号进行匹配，它们匹配成功才有可能是有效的括号
"""
def is_valid_brackets(s:str):
    if not s:
        return True
    stack = []
    for i in range(len(s)):
        if s[i] == '(' or s[i] == '[' or s[i] == '{':
            stack.append(s[i])
        elif s[i] == ')' and len(stack) > 0 and stack[-1] == '(' or \
            s[i] == ']' and len(stack) > 0 and stack[-1] == '[' or \
            s[i] == '}' and len(stack) > 0 and stack[-1] == '{':
                # stack = stack[:len(stack)-1]
                stack.pop()
        else:
            return False
    return len(stack) == 0

"""leetcode224 基本计算器
实现一个基本的计算器来计算一个简单的字符串表达式 s 的值。
输入：s = "1 + 1"
输出：2

输入：s = " 2-1 + 2 "
输出：3

输入：s = "(1+(4+5+2)-3)+(6+8)"
输出：23
"""
def calculate(s:str):
    stack = []
    # 为了方便计算，所有的操作都视为加法操作
    # 那么原先的减法操作就相当于是加一个负数
    # 默认都是正数
    sign = 1
    res = 0
    i = 0
    while i < len(s):
        if s[i] == ' ':
            i += 1
        elif s[i] == '+':
            sign = 1
            i += 1
        elif s[i] == '-':
            sign = -1
            i += 1
        elif s[i] == '(':
            stack.append(res)
            stack.append(sign)
            res = 0
            sign = 1
            i += 1
        elif s[i] == ')':
            res = stack.pop() * res + stack.pop()
            i += 1
        else: # 如果当前字符是数字的话
            # 那么把获取到的数累加到结果 res 上
            # 去查看当前字符的后一位是否存在
            # 如果操作并且后一位依旧是数字，那么就需要把后面的数字累加上来
            num = 0
            while i < len(s) and s[i].isdigit():
                num = num * 10 + int(s[i])
                i += 1
            res += sign * num
    return res

"""剑指offer 40 最小的k个数
"""
def get_least_numbers(arr:List[int],k:int):
    """使用快速排序实现

    Args:
        arr (List[int]): _description_
        k (int): _description_
    """
    def quick_sort(arr:List[int],left:int,right:int):
        if left >= right:
            return 
        i,j = left,right
        while i < j:
            while i < j and arr[j] >= arr[left]:
                j -= 1
            while i < j and arr[i] <= arr[left]:
                i += 1
            arr[i],arr[j] = arr[j],arr[i]
        arr[i],arr[left] = arr[left],arr[i]
        quick_sort(arr,left,i-1)
        quick_sort(arr,i+1,right)
    quick_sort(arr,0,len(arr)-1)
    return arr[:k]

"""leetcode 152 乘积最大子数组
给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
输入: [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。

输入: [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。

解题思路：
这道题是动态规划的题目，我们可以通过定义两个变量max_val和min_val来分别记录当前的最大值和最小值，然后通过遍历数组，
我们可以得到如下的状态转移方程：

• max_val 表示在数组 0-i 中，选取 nums[i] 时的最大乘积
• min_val 表示在数组 0-i 中，选取 nums[i] 时的最小乘积

max_val = max(max_val * nums[i],nums[i])
min_val = min(min_val * nums[i],nums[i])
res = max(res,max_val)

"""
def max_product(nums:List[int]):
    if not nums:
        return 0
    max_val = nums[0]
    min_val = nums[0]
    res = nums[0]
    for i in range(1,len(nums)):
        if nums[i] < 0:
            max_val,min_val = min_val,max_val
        max_val = max(max_val*nums[i],nums[i])
        min_val = min(min_val*nums[i],nums[i])
        res = max(res,max_val)
    return res

"""leetcode279 完全平方数
题目描述：给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。
你需要让组成和的完全平方数的个数最少。

输入: n = 12
输出: 3
解释: 12 = 4 + 4 + 4.

输入: n = 13
输出: 2
解释: 13 = 4 + 9.

解题思路：
这道题是动态规划的题目，我们可以通过定义一个dp数组来记录每一个数的最小的完全平方数的个数，然后通过遍历数组，
我们可以得到如下的状态转移方程： dp[i] = min(dp[i],dp[i-j*j]+1)
dp[i]定义：表示数字 i 最少可以由几个完全平方数组成
"""
def num_squares(n:int):
    dp = [0] * (n+1)
    for i in range(1,n+1):
        dp[i] = i # 最坏的情况就是由i个1组成，所以dp[i]=i
        j = 1
        while i - j*j >= 0:
            dp[i] = min(dp[i],dp[i-j*j]+1)
            j += 1
    return dp[n]

"""leetcode 3 无重复字符的最长子串
给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
输入: s = "abcabcbb"
输出: 3
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。

解题思路：
这道题是滑动窗口的题目，我们可以通过定义一个start变量来记录窗口的起始位置，然后通过遍历字符串，
我们可以得到如下的状态转移方程： res = max(res,i-start+1)
"""
def length_of_longest_substring(s:str):
    if not s:
        return 0
    start = 0
    res = 0
    map_dict = {}
    for i in range(len(s)):
        if s[i] in map_dict:
            start = max(start,map_dict[s[i]]+1)
        map_dict[s[i]] = i
        res = max(res,i-start+1)
    return res

"""leetcode 5 最长回文子串
给你一个字符串 s，找到 s 中最长的回文子串。
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。

输入：s = "cbbd"
输出："bb"

解题思路： 这道题是动态规划的题目，我们可以通过定义一个dp数组来记录每一个子串是否是回文串，然后通过遍历数组，
我们可以得到如下的状态转移方程：dp[i][j] = dp[i+1][j-1] and s[i] == s[j]
dp[i]定义：表示字符串 s 的第 i 到 j 个字符是否是回文串
"""
def longest_palindrome(s:str):
    if not s:
        return ""
    n = len(s)
    dp = [[False]*n for _ in range(n)]
    res = ""
    for i in range(n-1,-1,-1):
        for j in range(i,n):
            dp[i][j] = s[i] == s[j] and (j-i < 2 or dp[i+1][j-1])
            if dp[i][j] and j-i+1 > len(res):
                res = s[i:j+1]
    return res

"""leetcode 11 盛最多水的容器
给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点(i, ai) 。在坐标内画 n 条垂直线，
垂直线 i 的两个端点分别为(i, ai) 和(i, 0)。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

输入：[1,8,6,2,5,4,8,3,7]
输出：49

解题思路：这道题是对撞指针的题目，我们可以通过定义两个指针left和right来分别指向数组的头部和尾部，然后通过遍历数组，
我们可以得到如下的状态转移方程：res = max(res,min(height[left],height[right])*(right-left))
"""
def max_area(nums:List[int]):
    if not nums:
        return 0
    left , right = 0, len(nums) - 1
    res = 0
    while left < right:
        res = max(res,min(nums[left],nums[right])*(right-left))
        if nums[left] < nums[right]:
            left += 1
        else:
            right -= 1
    return res

"""leetcode 409 最长回文串
给定一个包含大小写字母的字符串，我们需要找到可以构造出的最长的回文串的长度。
输入: "abccccdd"
输出: 7
解释: 我们可以构造的最长的回文串是"dccaccd", 它的长度是 7。

解题思路：这道题是哈希表的题目，我们可以通过定义一个map字典来记录每一个字符出现的次数，然后通过遍历字典，
"""
def longest_palindrome(s:str):
    if not s:
        return 0
    map_dict = {}
    for i in range(len(s)):
        if s[i] in map_dict:
            map_dict[s[i]] += 1
        else:
            map_dict[s[i]] = 1
    res = 0
    for k,v in map_dict.items():
        res += v//2*2 # 取偶数
    return res if res == len(s) else res+1

"""剪绳子 
给你一根长度为n的绳子，请把绳子剪成m段(m,n都是整数，n>1并且m>1)，每段绳子的长度记为k[0],k[1],...,k[m]。
请问k[0]*k[1]*...*k[m]可能的最大乘积是多少？

输入：8
输出：18
解释：2*3*3=18

解题思路：这道题是动态规划的题目，我们可以通过定义一个dp数组来记录每一个数的最大的乘积，然后通过遍历数组，
我们可以得到如下的状态转移方程：dp[i] = max(dp[i],max(j*(i-j),j*dp[i-j]))
"""
def cutting_rope(n:int):
    if n < 2:
        return 0
    if n == 2:
        return 1
    if n == 3:
        return 2
    dp = [0] * (n+1)
    dp[1] = 1
    dp[2] = 2
    dp[3] = 3
    for i in range(4,n+1):
        for j in range(1,i//2+1):
            dp[i] = max(dp[i],max(j*(i-j),j*dp[i-j]))
    return dp[n]

""" leetcode plus one
输入：[1,2,3]
输出：[1,2,4]

输入：[9,9,9]
输出：[1,0,0,0]
"""
def plus_one(digits:List[int]):
    """可以额外使用数组

    Args:
        nums (List[int]): _description_
    """
    res = [d for d in digits]
    carry = 0
    digits_len = len(digits)
    res[digits_len-1] += 1
    for i in range(digits_len-1,-1,-1):
        res[i] = (digits[i] + carry) % 10
        carry =  (digits[i] + carry) // 10
        if not carry:
            return res
    if carry:
        res = [0] * (digits+1)
        res[0] = 1
    return res


# leetcode 215 数组中的第K个最大元素
class Solution:
    def findKthLargest(self,nums:List[int],k:int):
        lo, hi = 0, len(nums)-1
        k = len(nums) - k
        while lo < hi:
            p = self.partition(nums,lo,hi)
            if p < k:
                lo = p + 1 
            if p > k:
                hi = p - 1
            else:
                return nums[p]
        return -1 
    
    def partition(self,nums:List[int],low:int,high:int):
        pivot = nums[low]
        i,j = low+1,high
        while i <= j:
            while i < j and nums[i] <= pivot:
                i += 1 
            while i < j and nums[j] > pivot:
                j += 1 

            if i >= j:
                break
            nums[i],nums[j] = nums[j],nums[i]
        
        nums[low],nums[j] = nums[j],nums[low]
        return j 
        




if __name__ == "__main__":
    nums  = [1,2,3,4,5,6,7]
    target = 9 
    print(two_sum(nums, target))
    s = ["h","e","l","l","o"]
    s1 = ["w","e","l","l"]
    print(reverse_string(s))
    print(reverse_string(s1))
    x = -1233
    print(reverse_number(x))
    y = 1221
    print(is_palindrome(y))
    print(is_palindrome(x=10112))
    print(max_area(nums=[1,8,6,2,5,4,8,3,7]))
    print(maxArea(nums=[1,8,6,2,5,4,8,3,7]))
    print(three_sum(nums=[-1,-1,0,1,2,-4]))
    print(three_sum(nums=[-1,-1,-3,0,1,2,3,4,-2]))
    print(three_sum_closest(nums=[-1,-1,0,1,2,-4],target=0))
    print(vaild_brackets(s="()["))
    print(moveZeroes(nums=[-1,-1,0,1,0,2]))

    nums = [1,2,0,3,0,4,5]
    print(moveZeroes(nums))
    
    



