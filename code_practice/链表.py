from typing import Optional,List

class ListNode:
    def __init__(self,val=0,next=None):
        self.val = val
        self.next = next


# leetcode 链表移除倒数第k个节点
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(-1) # 虚拟头结点
        dummy.next = head
        n_1 = self.findFromEnd(dummy,n+1) # 找到第N+1个节点
        n_1.next = n_1.next.next  # 删除第N个节点
        return dummy.next

    def findFromEnd(self,head:ListNode,k:int): 
        """快慢指针，找到倒数第k个节点
        1. 首先让第一个节点走k步
        2. 然后让新的头结点和第一个节点同时往后走，直到第一个节点链表走到链表末尾，此时第二个节点为倒数第k个节点

        Args:
            head (ListNode): _description_
            k (int): _description_

        Returns:
            _type_: _description_
        """
        p1 = head
        for i  in  range(k):
            p1 = p1.next
        p2 = head 
        while p1 :
            p1 = p1.next 
            p2 = p2.next

        return p2
    

"""leetcode86 分隔链表
给你一个链表的头节点 head 和一个特定值 x ，请你对链表进行分隔，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前。

你应当 保留 两个分区中每个节点的初始相对位置。

输入：head = [1,4,3,2,5,2], x = 3
输出：[1,2,2,4,3,5]
"""
def partition(head:ListNode,x:int) -> ListNode:
    # 创建两个虚拟头结点
    dummy1 = ListNode(-1)
    dummy2 = ListNode(-1)
    p1 = dummy1
    p2 = dummy2

    while head:
        if head.val < x:
            p1.next = head
            p1 = p1.next
        else:
            p2.next = head
            p2 = p2.next
        head = head.next

    # 将两个链表连接起来    
    p1.next = dummy2.next
    p2.next = None
    return dummy1.next


"""leetcode 21 合并两个有序链表
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
"""
def mergeTwoLists(l1:ListNode,l2:ListNode) -> ListNode:
    dummy = ListNode(-1) # 虚拟头结点
    cur = dummy 
    while l1 and l2:
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 if l1 else l2 
    return dummy.next

"""leetcode 24 两两交换链表中的节点

"""
def swapPairs(head:ListNode) -> ListNode:
    dummy = ListNode(-1,next=head)
    cur = dummy
    while cur.next and cur.next.next:
        tmp = cur.next
        cur.next = cur.next.next
        tmp.next = cur.next.next
        cur.next.next = tmp
        cur = cur.next.next
    return dummy.next

def swapPairs2(head:ListNode) -> ListNode:
    """
    递归实现两两交换链表中的节点
    """
    if not head or not head.next:
        return head
    next_node = head.next
    head.next = swapPairs2(next_node.next)
    next_node.next = head 

    return next_node

"""leetcode 92 反转链表II
"""
def reverseBetween(head:ListNode,m:int,n:int) -> ListNode:
    dummy = ListNode(-1)
    dummy.next = head
    pre = dummy
    for _ in range(m-1):
        pre = pre.next
    cur = pre.next
    for _ in range(n-m):
        tmp = pre.next
        pre.next = cur.next
        cur.next = cur.next.next
        pre.next.next = tmp
    return dummy.next


"""leetcode 25 K个一组翻转链表"""
def reverseKGroup(head:ListNode,k:int) -> ListNode:
    dummy = ListNode(-1)
    dummy.next = head
    pre = dummy
    end = dummy
    while end.next:
        for i in range(k):
            if end:
                end = end.next
        if not end:
            break
        start = pre.next
        next = end.next
        end.next = None
        pre.next = reverse(start)
        start.next = next
        pre = start
        end = pre
    return dummy.next

def reverse(head:ListNode):
    """递归反转链表

    Args:
        head (ListNode): _description_
    """
    next = reverse(head.next)
    head.next.next = head
    head.next = None

    return next

def reverse(head:ListNode) -> ListNode:
    pre = None
    cur = head
    while cur:
        next = cur.next
        cur.next = pre
        pre = cur
        cur = next
    return pre

"""leetcode 61 旋转链表
"""
def rotateRight(head:ListNode,k:int) -> ListNode:
    if not head or not head.next:
        return head
    n = 1
    cur = head
    while cur.next:
        cur = cur.next
        n += 1
    cur.next = head
    k %= n
    for _ in range(n-k):
        cur = cur.next
    res = cur.next
    cur.next = None
    return res

"""leetcode 143 重排链表
"""
def reorderList(head:ListNode) -> None:
    if not head or not head.next:
        return head
    mid = findMid(head)
    l1 = head
    l2 = mid.next
    mid.next = None
    l2 = reverse(l2)
    merge(l1,l2)

def findMid(head:ListNode) -> ListNode:
    slow = head
    fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    return slow

def merge(l1:ListNode,l2:ListNode) -> None:
    while l1 and l2:
        tmp1 = l1.next
        tmp2 = l2.next
        l1.next = l2
        l1 = tmp1
        l2.next = l1
        l2 = tmp2

"""leetcode 234 回文链表
"""
def isPalindrome(head:ListNode) -> bool:
    if not head or not head.next:
        return True
    mid = findMid(head)
    l1 = head
    l2 = mid.next
    mid.next = None
    l2 = reverse(l2)
    while l1 and l2:
        if l1.val != l2.val:
            return False
        l1 = l1.next
        l2 = l2.next
    return True

"""leetcode 328 奇偶链表
"""
def oddEvenList(head:ListNode) -> ListNode:
    if not head or not head.next:
        return head
    odd = head
    even = head.next
    evenHead = even
    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next
    odd.next = evenHead
    return head

"""leetcode 19 删除链表的倒数第N个节点
"""
def removeNthFromEnd(head:ListNode,n:int) -> ListNode:
    dummy = ListNode(-1)
    dummy.next = head
    p1 = dummy
    p2 = dummy
    for _ in range(n):
        p1 = p1.next
    while p1.next:
        p1 = p1.next
        p2 = p2.next
    p2.next = p2.next.next
    return dummy.next

"""leetcode142 环形链表II 寻找环形链表起点
"""
def detectCycle(head:ListNode):
    slow = head
    fast = head
    while True:
        if not fast or not fast.next:
            return None
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    fast = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return slow

"""leetcode160 相交链表
"""
def getInsertersionNode(headA:ListNode,headB:ListNode) -> ListNode:
    """无论 A、B 两个链表是否有相交点，最终都会指向一个相同的节点，要么是它们的公共尾部，要么是 NULL。
    依托于上面这个结论，设置两个指针，pointA 和 pointB。
    让指针 pointA 和 pointB 分别指向链表 A 和链表 B 的头结点，之后两个指针分别同时以步幅为 1 的速度向链表的尾部遍历。
    当指针 pointA 遍历到链表 A 的尾节点时，此时指针 pointA 走了 a 个节点，将指针 pointA 指向链表 B 的头部，继续向后遍历，直至走到 c1，此时指针 pointA 总共走了 a + ( b - c ) 步。
    当指针 pointB 遍历到链表 B 的尾节点时，此时指针 pointB 走了 b 个节点，将指针 pointB 指向链表 A 的头部，继续向后遍历，直至走到 c1，此时指针 pointB 总共走了 b + ( a - c ) 步。
    
    根据数学知识，a + ( b - c ) = b + ( a - c ) 。
    如果 c > 0，表明两链表有公共尾部， c1 存在，两两链表同时到达 c1；
    如果 c = 0，表明两链表没有公共尾部，指针 pointA 和 pointB 都指向 NULL。

    这也从数学逻辑上证明了：只要二人有交集，就终会相遇。

    Args:
        headA (ListNode): _description_
        headB (ListNode): _description_

    Returns:
        ListNode: _description_
    """
    if not headA or not headB:
        return None
    p1 = headA
    p2 = headB
    while p1 != p2:
        p1 = p1.next if p1 else headB
        p2 = p2.next if p2 else headA
    return p1

"""leetcode328 奇偶链表
"""
def oddEvenList(head:ListNode) -> ListNode:
    if not head or not head.next:
        return head
    even = head.next # 偶数节点
    evenHead = even # 偶数节点头 
    while even and even.next:
        odd.next = even.next # 奇数节点
        odd = odd.next 
        even.next = odd.next
        even = even.next
    odd.next = evenHead # 奇数节点尾指向偶数节点头
    return head

"""leetcode 237 删除链表中的节点
"""
def deleteNode(node:ListNode) -> None:
    node.val = node.next.val
    node.next = node.next.next

"""leetcode 143 重排链表
"""
def reorderList(head:ListNode) -> None:
    if not head or not head.next:
        return head
    mid = findMid(head)
    l1 = head
    l2 = mid.next
    mid.next = None
    l2 = reverse(l2)
    merge(l1,l2)


if __name__ == "__main__":
    ListNode1 = ListNode(1)
    ListNode2 = ListNode(2)
    ListNode3 = ListNode(3)
    ListNode4 = ListNode(4)
    ListNode5 = ListNode(5)
    ListNode1.next = ListNode2
    ListNode2.next = ListNode3
    ListNode3.next = ListNode4
    ListNode4.next = ListNode5
    head = ListNode1
    print("head:",head)
    res = swapPairs2(head)
    print("res:",res)
