from typing import Optional,List

# 递归三要素
    # 1. 确定递归函数的参数和返回值
    # 2. 确定终⽌条件
    # 3. 确定单层递归的逻辑


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """中序递归遍历"""        
        result = self.traversal(root)
        return result

    def traversal(self, cur: TreeNode):
        if not cur:   # 递归终止条件
            return []    # 函数返回类型

        left = self.traversal(cur.left)
        right = self.traversal(cur.right)

        result = left + [cur.val] + right  # 单层递归逻辑

        return result


class Solution:
    def preTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """前序递归遍历"""        
        result = self.traversal(root)
        return result

    def traversal(self, cur: TreeNode):
        if not cur:
            return []

        left = self.traversal(cur.left)
        right = self.traversal(cur.right)

        result = [cur.val] + left + right

        return result
    
class Solution:
    def postTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """后序递归遍历"""        
        result = self.traversal(root)
        return result

    def traversal(self, cur: TreeNode):
        if not cur:
            return []

        left = self.traversal(cur.left)
        right = self.traversal(cur.right)

        result = left + right + [cur.val]

        return result
    


# N叉树递归遍历

class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class Solution:
    def postorder(self, root: Node) -> List[int]:
        """ N叉树后序遍历

        Args:
            root (Node): _description_

        Returns:
            List[int]: _description_
        """
        result = self.traversal(root)
        return result

    def traversal(self,cur: Node):
        if not cur:
            return []

        child_res = []
        for child in cur.children:
            child_res.append(self.traversal(child))
        
        result = child_res + [cur.val]

        return result
    

# 203 移除链表元素 
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        result = self.traversal(head,val)
        return result

    def traversal(self,head:Optional[ListNode],val:int):
        if not head:
            return head 
        
        head.next = self.traversal(head.next,val)

        if head.val == val:
            return head.next
        else:
            return head
        
# 翻转列表
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:  # 边缘条件
            return head 
        last = self.reverseList(head.next) 
        head.next.next = head    # 反转链表
        head.next = None    # 头结点指向空
        return last


# 224 翻转二叉树
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
        
        # 后序遍历   分解问题求解
        left = self.invertTree(root.left) 
        right = self.invertTree(root.right)
        root.left, root.right = right, left
        
        return root
    
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
        
        # 前序遍历   遍历二叉树求解
        root.left, root.right = root.right, root.left
        self.invertTree(root.left) 
        self.invertTree(root.right)
        
        return root