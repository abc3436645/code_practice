class listNode:
    def __init__(self,val):
        self.val = val
        self.next = None

###递归反转链表
class linkList:
    def reverse(self,head:listNode):
        if head.next is None:
            return head
        last = self.reverse(head.next) ##反转之后head节点到达了链表的结尾
        head.next.next = head   ##将第二个节点指向第一个节点
        head.next = None        ##head节点指向None  

        return last
    def reverseN(self, head, n):
        if n == 1:
            self.next_of_last = head.next
            return head
        if not head or not head.next: 
            return head
        last = self.reverseN(head.next, n - 1)
        head.next.next = head
        head.next = self.next_of_last
        return last

    def reverseBetween(self, head: listNode, m: int, n: int):
        if m == 1: 
            last = self.reverseN(head, n) 
            return last
        head.next  = self.reverseBetween(head.next, m - 1, n - 1)
        return head


    def linkedList_loop(self,cur:listNode):  ##链表遍历
        ele_list = []
        while cur is not None:
            ele_list.append(cur.val)
            cur = cur.next
        return ele_list        

if __name__ == "__main__":
    nums_list = [1,2,3,4,5,6,7,8,9]
    head = listNode(val='head')
    cur = head
    for num in nums_list:
        num_node = listNode(num)
        cur.next = num_node
        cur = cur.next
    
    linklist = linkList()
    ele_list = linklist.linkedList_loop(head)
    print(ele_list)
    
    reverse_linkedlist_head = linklist.reverse(head)
    reverse_ele_list = linklist.linkedList_loop(reverse_linkedlist_head)
    print('反转之后的节点:',reverse_ele_list)

    reverseN_head = linklist.reverseN(head,4)
    reverseN_list = linklist.linkedList_loop(reverseN_head)
    print('反转前N个节点之后:',reverseN_list)
    print('head:',ele_list)
