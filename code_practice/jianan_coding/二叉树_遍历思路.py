from typing import List


# ListNode 
class ListNode:
    def __init__(self, val,next):
        self.val = val
        self.next = next

# 二叉树
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

"""
二叉树解题的思维模式分两类：

1、是否可以通过遍历一遍二叉树得到答案？如果可以，用一个 traverse 函数配合外部变量来实现，这叫「遍历」的思维模式。

2、是否可以定义一个递归函数，通过子问题（子树）的答案推导出原问题的答案？如果可以，写出这个递归函数的定义，并充分利用这个函数的返回值，这叫「分解问题」的思维模式。

无论使用哪种思维模式，你都需要思考：

如果单独抽出一个二叉树节点，它需要做什么事情？需要在什么时候（前/中/后序位置）做？其他的节点不用你操心，递归函数会帮你在所有节点上执行相同的操作。
"""

"""递归三要素
1.确定递归函数的参数和返回值： 确定哪些参数是递归的过程中需要处理的，那么就在递归函数里加上这个参数， 并且还要明确每次递归的返回值是什么进而确定递归函数的返回类型。
2.确定终止条件： 写完了递归算法, 运行的时候，经常会遇到栈溢出的错误，就是没写终止条件或者终止条件写的不对，操作系统也是用一个栈的结构来保存每一层递归的信息，如果递归没有终止，操作系统的内存栈必然就会溢出。
3.确定单层递归的逻辑： 确定每一层递归需要处理的信息。在这里也就会重复调用自己来实现递归的过程。

"""


# leetcode 124 二叉树最大路径和
class Solution:
    def maxPathSum(self, root: TreeNode):
        self.res = float("-inf")
        self.oneside(root)
        return self.res

    def oneside(self, root: TreeNode):
        """
        func定义：以root为根节点的最大路径
        """
        if not root:
            return 0
        left = max(0, self.oneside(root.left))
        right = max(0, self.oneside(root.right))
        self.res = max( 
            self.res, left + right + root.val
        )  # 利用左右子树结果顺带求出最大路径

        return max(left, right) + root.val


# leetcode 104 二叉树最大深度
class Solution:
    def maxDepth(self, root: TreeNode):
        self.res = 0
        self.depth = 0
        self.traverse(root)

        return self.res

    def traverse(self, root: TreeNode):
        """
        遍历思维解法：用一个 traverse 函数配合外部变量来实现
        """
        if not root:
            return
        self.depth += 1
        self.res = max(self.res, self.depth)
        self.traverse(root.left)
        self.traverse(root.right)
        self.depth -= 1


class Solution:
    def maxDepth(self, root: TreeNode):
        if not root:
            return 0
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)

        return max(left_depth, right_depth) + 1


# leetcode 111 二叉树最小深度
class Solution:
    def minDepth(self, root: TreeNode):
        self.res = float("inf")
        self.depth = 0
        self.traverse(root)

        return self.res if self.res != float("inf") else 0

    def traverse(self, root: TreeNode):
        """
        遍历思维：使用traverse函数配合外部变量来实现
        """
        if not root:
            return
        self.depth += 1
        self.res = min(self.res, self.depth)
        self.traverse(root.left)
        self.traverse(root.right)
        self.depth -= 1


class Solution:
    def minDepth(self, root: TreeNode):
        if not root:
            return 0

        if root.left is None and root.right is not None:
            return self.minDepth(root.right) + 1
        if root.right is None and root.left is not None:
            return self.minDepth(root.left) + 1
        left = self.minDepth(root.left)
        right = self.minDepth(root.right)

        res = min(left, right) + 1

        return res


# leetcode 543 二叉树的直径
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode):
        # 每一条二叉树的「直径」长度，就是一个节点的左右子树的最大深度之和
        self.max_diameter = 0
        self.traverse(root)

        return self.max_diameter

    def maxDepth(self, root):
        if not root:
            return 0
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)

        max_depth = max(left, right) + 1

        return max_depth

    def traverse(self, root: TreeNode):
        if not root:
            return
        leftMax = self.maxDepth(root.left)
        rightMax = self.maxDepth(root.right)
        myDiameter = leftMax + rightMax

        self.max_diameter = max(self.max_diameter, myDiameter)
        self.traverse(root.left)
        self.traverse(root.right)


class Solution:
    def diameterOfBinaryTree(self, root: TreeNode):
        self.max_diameter = 0
        self.maxDepth(root)

        return self.max_diameter

    def maxDepth(self, root: TreeNode):
        """
        以root为根节点的最大高度
        """
        if not root:
            return 0
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        myDiameter = left + right

        self.max_diameter = max(self.max_diameter, myDiameter)

        return max(left, right) + 1


from collections import deque


# 二叉树层序遍历
def levelTraverse(root: TreeNode):
    if not root:
        return
    dq = deque()
    dq.append(root)
    res = []
    while dq:
        size = len(dq)
        for _ in range(size):
            node = dq.pop()
            res.append(node.val)
            if node.left:
                dq.append(node.left)
            if node.right:
                dq.append(node.right)


# leetcode 257 二叉树所有路径
class Solution:
    def binaryTreePaths(self, root: TreeNode):
        self.res = []
        self.path = []
        self.traverse(root)
        return self.res

    def traverse(self, root: TreeNode):
        if not root:
            return

        self.path.append(str(root.val))
        if not root.left and not root.right:  # 到达叶子节点
            path = "->".join(self.path)
            self.res.append(path[:])
        self.traverse(root.left)
        self.traverse(root.right)
        self.path.pop()


# leetocode 129 求根节点到叶节点数字之和
class Solution:
    def sumNumbers(self, root: TreeNode):
        self.res = 0
        self.path = ""
        self.traverse(root)
        return self.res

    def traverse(self, root: TreeNode):
        if not root:
            return
        self.path += str(root.val)
        if not root.left and not root.right:
            self.res += int(self.path)
        self.traverse(root.left)
        self.traverse(root.right)
        self.path = self.path[:-1]


# leetcode 199 二叉树的右视图
class Solution:
    def rightSideView(self, root: TreeNode):
        """
        遍历问题思路：使用traverse函数配合外部变量来实现
        """
        self.res = []
        self.depth = 0
        self.traverse(root)

        return self.res

    def traverse(self, root: TreeNode):
        if not root:
            return
        self.depth += 1
        if len(self.res) < self.depth:
            self.res.append(root.val)
        self.traverse(root.right)
        self.traverse(root.left)
        self.depth -= 1


class Solution:
    def rightSideView(self, root: TreeNode):
        if not root:
            return []
        return self.traverse(root)

    def traverse(self, root: TreeNode):
        res = []
        dq = deque()
        dq.append(root)
        while dq:
            size = len(dq)
            last = dq[0]
            for _ in range(size):
                node = dq.popleft()
                if node.right:
                    dq.append(node.right)
                if node.left:
                    dq.append(node.left)
            res.append(last.val)
        return res


# leetcode 298 二叉树最长连续序列
class Solution:
    def longestConsecutive(self, root: TreeNode):
        self.max_len = 0
        self.traverse(root, 1, float("-inf"))

        return self.max_len

    def traverse(self, root: TreeNode, length: int, parent_val: int):
        if not root:
            return
        if root.val == parent_val + 1:
            length += 1
        else:
            length = 1
        self.max_len = max(self.max_len, length)
        self.traverse(root.left, length, root.val)
        self.traverse(root.right, length, root.val)


# leetcode 988 从叶结点开始的最小字符串
class Solution:
    def smallestFromLeaf(self, root: TreeNode):
        self.res = None
        self.path = []
        self.traverse(root)

        return self.res

    def traverse(self, root: TreeNode):
        if not root:
            return
        if not root.left and not root.right:
            self.path.append(chr(ord("a") + root.val))
            s = "".join(self.path[::-1])
            if not self.res or self.res > s:
                self.res = s
            self.path.pop()
        self.path.append(chr(ord("a") + root.val))
        self.traverse(root.left)
        self.traverse(root.right)
        self.path.pop()


# leetcode 1457 二叉树中的伪回文路径
# 如果一组数字中，只有最多一个数字出现的次数为奇数，剩余数字的出现次数均为偶数，那么这组数字可以组成一个回文串
#
class Solution:
    def pseudoPalindromicPaths(self, root: TreeNode):
        self.count = [0] * 10  # 统计出现的次数
        self.res = 0
        self.traverse(root)
        return self.res

    def traverse(self, root: TreeNode):
        if not root:
            return
        if not root.left and not root.right:
            self.count[root.val] += 1
            odd = sum([1 for n in self.count if n % 2 == 1])
            if odd <= 1:
                self.res += 1
            self.count[root.val] -= 1
        self.count[root.val] += 1
        self.traverse(root.left)
        self.traverse(root.right)
        self.count[root.val] -= 1


# leecode 270  最接近的二叉搜索树值
class Solution:
    def closestValue(self, root: TreeNode, target: float):
        self.res = float("inf")
        self.traverse(root, target)

        return self.res

    def traverse(self, root: TreeNode, target: float):
        if not root:
            return
        if abs(root.val - target) < abs(self.res - target):
            self.res = root.val
        if root.val > target:
            self.traverse(root.left)
        else:
            self.traverse(root.right)


# leetcode 404 左叶子之和
class Solution:
    def sumOfLeftLeaves(self, root: TreeNode):
        self.res = 0
        self.traverse(root)

        return self.res

    def traverse(self, root: TreeNode):
        if not root:
            return
        if root.left and not root.left.left and not root.left.right:
            self.res += root.left.val
        self.traverse(root.left)
        self.traverse(root.right)


# leetcode 617 合并二叉树
class Solution:
    def mergeTrees(self, root1: TreeNode, root2: TreeNode):
        if not root1:
            return root2
        if not root2:
            return root1
        root1.val += root2.val
        root1.left = self.mergeTrees(root1.left,root2.left)
        root1.right = self.mergeTrees(root1.right,root2.right)

        return root1

# leetcode 623 在二叉树中增加一行
class Solution:
    def addOneRow(self, root: TreeNode, val: int, depth: int):
        self.targetVal = val
        self.targetDepth = depth
        self.curDepth = 0
        if self.targetDepth == 1:
            newRoot = TreeNode(val=self.targetVal)
            newRoot.left = root
            
            return newRoot
        self.traverse(root)

        return root
    
    def traverse(self,root:TreeNode):
        if not root:
            return 
        self.curDepth += 1
        if self.curDepth = self.targetDepth -1 :
            newLeft = TreeNode(val=self.targetVal)
            newRight = TreeNode(val=self.targetVal)
            newLeft.left = root.left
            newRight.right = root.right
            root.left = newLeft
            root.right = newRight
        self.traverse(root.left)
        self.traverse(root.right)
        self.curDepth -= 1
    

# leetcode 971 翻转二叉树
class Solution:
    def flipMatchVoyage(self,root:TreeNode,voage:List[int]):
        self.voage = voage
        self.res = []
        self.canFlip = True
        self.i = 0
        
        self.traverse(root)

        if self.canFlip:
            return self.res
        return [-1]

    def traverse(self,root:TreeNode):
        if not root or not self.canFlip:
            return 
        if root.val!= self.voage[self.i]:
            # 节点的 val 对不上，无解
            self.canFlip = False
            return
        self.i += 1 
        if root.left and root.left.val!= self.voage[self.i]:
            root.left,root.right = root.right,root.left # 翻转
            self.res.append(root.val)

        self.traverse(root.left)
        self.traverse(root.right)
        

# leetcode 987 二叉树垂序遍历
class Triple:
    def __init__(self,node:int,row:int,col:int):
        self.node = node
        self.row = row
        self.col = col

class Solution:
    def verticalTraversal(self,root:TreeNode):
        self.nodes = []
        self.traverse(root,0,0)
        
        # 按照 col 从小到大排序，col 相同的话按 row 从小到大排序，
        # 如果 col 和 row 都相同，按照 node.val 从小到大排序。
        self.nodes.sort(key=lambda x: (x.col, x.row, x.node.val))
        res = []
        preCol = float('-inf')
        for i in range(len(self.nodes)):
            cur = self.nodes[i]
            if cur.col != preCol:
                # 开始记录新的一列
                res.append([])
                preCol = cur.col
            res[-1].append(cur.node.val)

        return res
    
    def traverse(self,root:TreeNode,row:int,col:int):
        if not root:
            return
        self.nodes.append(Triple(root,row,col))
        self.traverse(root,row,col-1)
        self.traverse(root,row,col+1)


# leetcode 993 二叉树的堂兄弟
# 如果二叉树的两个节点深度相同，但 父节点不同，则它们是一对堂兄弟节点。
class Solution:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.depthX = 0
        self.depthY = 0
        self.parentX = 0
        self.parentY = 0
        self.depth = 0


    def isCousins(self,root:TreeNode,x:int,y:int):
        self.x = x
        self.y = y
        self.traverse(root,None)
        if self.parentX != self.parentY and self.depthX == self.depthY:
            return True
        return False

    def traverse(self,root:TreeNode,parent:TreeNode):
        if not root:
            return 
        self.depth += 1 
        if root.val == self.x:
            self.depthX = self.depth
            self.parentX = parent
        if root.val == self.y:
            self.depthY = self.depth
            self.parentY = parent
        self.traverse(root.left,root)
        self.traverse(root.right,root)
        self.depth -= 1
        
        

# leetocde 1315 祖父节点值为偶数的节点和
# 一个节点的祖父节点是指该节点的父节点的父节点
class Solution:
    def sumEvenGrandparent(self,root:TreeNode):
        self.res = 0
        self.traverse(root)
        
        return self.res
    
    def traverse(self,root:TreeNode):
        if not root:
            return 
        if root.val % 2 ==0:
            if root.left:
                if root.left.left:
                    self.res += root.left.left.val
                if root.left.right:
                    self.res += root.left.right.val

            if root.right:
                if root.right.left:
                    self.res += root.right.left.val
                if root.right.right:
                    self.res += root.right.right.val
        self.traverse(root.left)
        self.traverse(root.right)

# leetocde 1448 统计二叉树中好节点的数目
# 「好节点」X 定义为：从根到该节点 X 所经过的节点中，没有任何节点的值大于 X 的值。
class Solution:
    def goodNodes(self,root:TreeNode):
        self.res = 0
        self.traverse(root,root.val)
        
        return self.res
    
    def traverse(self,root:TreeNode,pathMax:int):
        if not root:
            return 
        if root.val >= pathMax:
            self.res += 1 
        pathMax = max(pathMax,root.val)
        self.traverse(root.left,pathMax)
        self.traverse(root.right,pathMax)

# leetcode 1469 寻找所有的独生节点
# 二叉树中，如果一个节点是其父节点的唯一子节点，则称这样的节点为独生节点。二叉树的根节点不会是独生节点，因为它没有父节点。
class Solution:
    def getLonelyNodes(self,root:TreeNode):
        self.res = []
        self.traverse(root) 
        
        return self.res

    def traverse(self,root:TreeNode):
        if not root:
            return
        if root.left and not root.right:
            self.res.append(root.left.val)
        if root.right and not root.left:
            self.res.append(root.right.val)

        traverse(root.left)
        traverse(root.right)

# leetcode 1602 找到二叉树中最近的右侧节点
# 给定一棵二叉树的根节点 root 和树中的一个节点 u，返回与 u 所在层中距离最近的右侧节点，当 u 是所在层中最右侧的节点，返回 null。
class Solution:
    def findNearestRightNode(self,root:TreeNode,u:TreeNode):
        self.targetDepth = 0
        self.res = 0
        self.depth = 0
        self.traverse(root,u.val)
        
        return self.res

    def traverse(self,root,targetVal:int):
        if not root:
            return  
        self.depth += 1
        if root.val == targetVal:
            self.targetDepth = self.depth 
        elif self.depth == self.targetDepth:
            self.res = root
            return
        self.traverse(root.left, targetVal)
        self.traverse(root.right,targetVal)
        self.depth -= 1

# leetcode 513 找树左下角的值
# 给定一个二叉树的根节点 root，请找出该二叉树的最底层最左边节点的值。假设二叉树中至少有一个节点。
class Solution:
    def findBottomLeftValue(self,root:TreeNode):
        self.res = None
        self.depth = 0
        self.max_depth = 0
        self.traverse(root)

        return self.res.val

    def traverse(self,root:TreeNode):
        if not root:
            return
        self.depth += 1
        if self.depth > self.max_depth:
            # 到最大深度时第一次遇到的节点就是左下角的节点
            self.max_depth = self.depth
            self.res = root
        self.traverse(root.left)
        self.traverse(root.right)
        self.depth -= 1


# leetcode 572 另一棵树的子树
# 给你两棵二叉树 root 和 subRoot。检验 root 中是否包含和 subRoot 具有相同结构和节点值的子树。如果存在，返回 true；否则，返回 false。
class Solution:
    def isSubtree(self,root:TreeNode,subRoot:TreeNode):
        if not root:
            return not subRoot
        if self.isSameTree(root,subRoot):
            return True
        return self.isSubtree(root.left,subRoot) or self.isSubtree(root.right, subRoot)
    
    def isSameTree(self,root1:TreeNode,root2:TreeNode):
        if not root1 and not root2:
            return True
        if not root1 or not root2:
            return False
        if root1.val != root2.val:
            return False
        
        return self.isSameTree(root1.left, root2.left) and self.isSameTree(root1.right,root2.right)


# leetcode 1367 二叉树中的列表
# 给你一棵以 root 为根的二叉树和一个 head 为第一个节点的链表。如果在二叉树中，存在一条一直向下的路径，且每个点的数值恰好一一对应以 head 为首的链表中每个节点的值，那么请你返回 True，否则返回 False。
class Solution:
    def isSubPath(self,head:ListNode,root:TreeNode):
        if not head:
            return True
        if not root:
            return False
        if head.val == root.val:
            if self.check(head,root):
                return True
        return self.isSubPath(head,root.left) or self.isSubPath(head,root.right)
    

    
    def check(self,head:ListNode,root:TreeNode):
        if not head:
            return True
        if not root:
            return False
        
        if head.val == root.val:
            return self.check(head.next,root.left) or self.check(head.next,root.right)

        return False

# leetcode 437 路径总和 III
# 给定一个二叉树的根节点 root，和一个整数 targetSum，求该二叉树里节点值之和等于 targetSum 的路径的数目。
# 路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
class Solution:
    def __init__(self):
        self.res = 0
        self.preSumCount = {} # 前缀和 定义：从二叉树的根节点开始，路径和为 pathSum 的路径有 preSumCount.get(pathSum) 个
        self.path_sum = 0
        self.targetSum = 0

    def pathSum(self,root:TreeNode,targetSum:int):
        self.targetSum = targetSum
        self.preSumCount[0] = 1
        self.traverse(root)
        
        return self.res

    def traverse(self,root:TreeNode):
        if not root:
            return 
        self.path_sum += root.val
        self.res += self.preSumCount.get(self.path_sum - self.targetSum,0)
        self.preSumCount[self.path_sum] = self.preSumCount.get(self.path_sum,0) + 1
        self.traverse(root.left)
        self.traverse(root.right)
        self.preSumCount[self.path_sum] =  self.preSumCount[self.path_sum] - 1
        self.pathSum -= root.val
