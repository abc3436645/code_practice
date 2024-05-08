from collections import deque

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children

# 二叉树层次遍历递归实现
class LevelTreeTravel():
    def level_order(self,root:TreeNode,res:list,level:int):
        if root is None:
            return []
        if len(res) < level+1:
            res.append([])
        res[level].append(root.val)
        self.level_order(root.left,res,level+1)
        self.level_order(root.right,res,level+1)
        return res
    
# 二叉树层次遍历迭代实现
class LevelTreeTravel():
    def level_order_itertor(self,root:TreeNode):
        if root is None:
            return []
        res = []
        deq = deque()
        deq.append(root)
        while deq:
            size = len(deq)
            level = []
            for i in range(size):
                node = deq.popleft()
                level.append(node.val)
                if node.left:
                    deq.append(node.left)
                if node.right:
                    deq.append(node.right)
            res.append(level)
        return res


###前序遍历
class Tree_Travel():
    ###前序遍历
    def prorder(self,root:TreeNode,res:list):
        if root is None:
            return []
        res.append(root.val)
        self.prorder(root.left)
        self.prorder(root.right)
        return res
    def prorder_itertor(self,root:TreeNode):
        """
        DFS前序遍历使用栈数据结构来访问
        """
        if root is None:
            return []
        res = []
        stack = []
        stack.append(root)
        while stack:
            node = stack.pop()
            if node:
                res.append(node.val)
            else:
                continue
            stack.append(node.right)
            stack.append(node.left)

        return  res

    def order(self,root:TreeNode,res:list):
        if root is None:
            return []
        self.order(root.left)
        res.append(root.val)
        self.order(root.right)

    def order_itertor(self,root:TreeNode):
        stack = []
        cur = root
        res = []
        while cur:
            if cur is not None:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                res.append(cur.val)
                cur = cur.right
        return res

    def prorder(self,root:TreeNode,res:list):
        if root is None:
            return []
        self.prorder(root.left)
        self.prorder(root.right)
        res.append(root.val)

    def prorder_itertor(self,root:TreeNode):
        stack = []
        res = []
        stack.append(root)
        while stack:
            node = stack.pop()
            if node:
                res.append(node.val)
            else:
                continue
            stack.append(node.left)
            stack.append(node.right)
        res = reversed(res)
        return res

####双色迭代法
## 未访问的节点为White,访问后的节点为Gray
def prorder_itertor(root:TreeNode):
    WHITE,GRAY = 0,1
    stack = [(root,WHITE)]
    res = []
    while stack:
        node,color = stack.pop()
        if node is None:
            continue
        if color == WHITE:
            stack.append((node.right,WHITE))
            stack.append((node.left,WHITE))
            stack.append((node,GRAY))
        elif color == GRAY:
            res.append(node.val)
    return res

def order_itertor(root:TreeNode):
    WHITE,GRAY = 0,1
    stack = [(root,WHITE)]
    res = []
    while stack:
        node,color = stack.pop()
        if node is None:
            continue
        if color == WHITE:
            stack.append((node.right,WHITE))
            stack.append((node,GRAY))
            stack.append((node.left,WHITE))

        elif color == GRAY:
            res.append(node.val)
    return res

def prerder_itertor(root:TreeNode):
    WHITE,GRAY = 0,1
    stack = [(root,WHITE)]
    res = []
    while stack:
        node,color = stack.pop()
        if node is None:
            continue
        if color == WHITE:
            stack.append((node,GRAY))
            stack.append((node.right,WHITE))
            stack.append((node.left,WHITE))
        elif color == GRAY:
            res.append(node.val)
    return res



###101对称二叉树
class Solution():
    """对称二叉树，对2边子树的base case进行判断，然后分别判断左右子树"""
    def compare(self,root:TreeNode):
        if root is None:
            return True
        return self.is_same(root.left,root.right)
    def is_same(self,left:TreeNode,right:TreeNode):
        if left is None and right is None:
            return True
        elif left is None and right is not None:
            return False
        elif left is not None and right is None:
            return False
        elif left.val != right.val:
            return False
        else:
            outside = self.is_same(left.left,right.right)
            inside = self.is_same(left.right,right.left)
        return True if outside and inside else False

class Solution:
    """迭代法"""
    def is_same(self,root:TreeNode):
        """对称二叉树判断思想，对root节点的左右子节点分别加入到容器(队列，数组，栈都可以)中，然后取出分别对比，判断为空情况下的结果，以及其他情况。
        不相等则返回Flase。"""
        from collections import deque

        if root is None:
            return True
        deq = deque()
        deq.append(root.left)
        deq.append(root.right)
        while deq:
            left_node = deq.popleft()
            right_node = deq.popleft()
            if left_node is None and right_node is None:
                continue
            if left_node is None or right_node is None or left_node.val != right_node.val:
                return False
            deq.append(left_node.left)
            deq.append(right_node.right)
            deq.append(left_node.right)
            deq.append(right_node.left)

        return True


###104二叉树最大深度
class Solution():
    def __init__(self) -> None:
        self.res = 0
        self.depth = 0
    def max_depth(self,root:TreeNode):
        """
        递归法，最大深度为max(左子树,右子树)+1
        """
        if root is None:
            return 0
        left = self.max_depth(root.left)
        right = self.max_depth(root.right)
        return max(left,right)+1
    
    def max_depth_2(self,root:TreeNode):
        """遍历思维，使用前序遍历，记录当前深度，当遍历到叶子节点时，更新最大深度

        Args:
            root (TreeNode): _description_
        """
        if not root:
            return 
        if not root.left and not root.right:
            self.res = max(self.res,self.depth)
        self.depth += 1
        self.max_depth_2(root.left)
        self.max_depth_2(root.right)
        self.depth -= 1

        return self.res

    def max_depth_itertor(self,root:TreeNode):
        """迭代法，二叉树的层数等于二叉树的最大深度，因此使用层数遍历即可。遍历完每层，depth+1，使用BFS，数据结构使用队列deque"""
        from collections import deque
        
        deq = deque()
        deq.append(root)
        depth = 0
        while deq:
            size = len(deq)
            depth += 1
            for i in range(size):
                node = deq.popleft()
                if node.left:
                    deq.append(node.left)
                if node.right:
                    deq.append(node.right)
        return depth


###559 N叉树的最大深度
class Solution:
    def max_depth_N(self,root:Node):
        if root is None:
            return 0
        depth = 0
        for child in root.children:
            depth = max(depth,self.max_depth_N(child))
        
        return depth+1

    def max_depth_N_itertor(self,root:Node):
        from collections import deque

        if root is None:
            return 0
        deq = deque()
        deq.append(root)
        depth = 0
        while deq:
            size = len(deq)
            depth += 1
            for i in range(size):
                node = deq.popleft()
                for child in node:
                    if child:
                        deq.append(child)
        return depth

####111二叉树的最小深度
class Solution:
    """"
    递归解法
    注意的是：当树的左子树或者右子树为空时，二叉树的最小深度不是0，而是不为空的子树的最小深度
    """
    def min_depth(self,root:TreeNode):
        if root is None:
            return 0
        left_depth = self.min_depth(root.left)
        right_depth = self.min_depth(root.right)
        if root.left is None and root.right is not None:
            return 1+right_depth
        if root.left is not None and root.right is None:
            return 1+left_depth
        res = min(left_depth,right_depth)+1
        return res

    def min_depth_itertor(self,root:TreeNode):
        """
        迭代法:使用层次遍历的方式解决,当左右孩子都是空的情况下，才说明遍历到最低点了，如果其中孩子不为空则不是最低点
        """
        from collections import deque

        if root is None:
            return 0
        deq = deque()
        depth = 0
        deq.append(root)
        while deque:
            size = len(deque)
            depth += 1
            flag = 0
            for i in range(size):
                node = deque.popleft()
                if node.left:
                    deq.append(node.left)
                if node.right:
                    deq.append(node.right)
                if node.left is None and node.right is None:
                    flag = 1
                    break
        if flag == 1:
            pass
        return depth


###222完全二叉树的节点个数
class Solution:
    def get_tree_nodes_nums(self,root:TreeNode):

        """递归方法，顺序是后序遍历,获取左子树的节点，再获取右子树的节点数，最后加上根节点"""
        if root is None: ### base case
            return 0 
        left_nums = self.get_tree_nodes_nums(root.left)
        right_nums = self.get_tree_nodes_nums(root.right)
        tree_nums = left_nums+right_nums+1
        return tree_nums
    
    def get_tree_nodes_nums_itertor(self,root:TreeNode):
        """迭代方法，思路节点分别进入容器，统计节点数量即可"""
        from collections import deque
        if root is None:
            return 0
        res = 0
        deq = deque()
        while deq:
            size = len(deq)
            for i in range(size):
                node = deq.popleft()
                res += 1
                if node.left:
                    deq.append(node.left)
                if node.right:
                    deq.append(node.right)

        return res

####110平衡二叉树
class Solution:
    def is_balanced(self,root:TreeNode):
        """递归，根据左右子树的高度进行判断，如果是二叉树则左右子树高度差不超过1，如果是平衡二叉树，返回平衡二叉树高度"""
        return False if self.get_depth(root) == -1 else True
 
    def get_depth(self,root:TreeNode):
        if root is None:
            return 0
        left_depth = self.get_depth(root.left)
        right_depth = self.get_depth(root.right)
        if left_depth == -1:
            return -1
        if right_depth == -1:
            return -1 
        return -1 if abs(left_depth-right_depth)>1 else 1+max(left_depth,right_depth)


####257二叉树的所有路径
class Solution:
    def traversal(self,root:TreeNode,path:list,res:list):
        path.append(root.val)
        if root.left is None and root.right is None:  ###代表path路径到达叶子节点
            s_path = ''
            for i in range(len(path)-1):
                s_path += str(path[i])
                s_path += '->'
            s_path += str(path[len(path)-1])
            res.append(s_path)
            return 
        
        if root.left:
            self.traversal(root.left)
            path.pop()  ##回溯
        if root.right:
            self.traversal(root.right)
            path.pop()  ##回溯

    def binary_tree_paths(self,root:TreeNode):
        """
        递归加回溯
        求根节点到叶子节点的路径，所以需要前序遍历，需要回溯才能访问其他路径
        """
        res = []    
        path = []        
        if root is None:
            return res
        self.traversal(root,path,res)
        return res

    
class Solution:
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        def construct_paths(root, path):
            if root:
                path += str(root.val)
                if not root.left and not root.right:  # 当前节点是叶子节点
                    paths.append(path)  # 把路径加入到答案中
                else:
                    path += '->'  # 当前节点不是叶子节点，继续递归遍历
                    construct_paths(root.left, path)
                    construct_paths(root.right, path)

        paths = []
        construct_paths(root, '')
        return paths
    
class Solution:
    def binaryTreePaths(self, root: TreeNode):
        paths = list()
        if not root:
            return paths

        node_queue = deque([root])
        path_queue = deque([str(root.val)])

        while node_queue:
            node = node_queue.popleft()
            path = path_queue.popleft()

            if not node.left and not node.right:
                paths.append(path)
            else:
                if node.left:
                    node_queue.append(node.left)
                    path_queue.append(path + '->' + str(node.left.val))
                
                if node.right:
                    node_queue.append(node.right)
                    path_queue.append(path + '->' + str(node.right.val))
        return paths

    def binary_tree_paths_itertor(self,root:TreeNode):
        tree_stack = [] ###保存树的遍历节点
        path_stack = []
        res = []
        if root is None:
            return []
        tree_stack.append(root)
        path_stack.append(str(root.val))
        while tree_stack:
            node = tree_stack.pop()
            path = path_stack.pop()
            if node.left is None and node.right is None:
                res.append(path)
            if node.right:
                tree_stack.append(node.right)
                path_stack.append(path+'->'+str(node.right.val))
            if node.left:
                tree_stack.append(node.left)
                path_stack.append(path+'->'+str(node.left.val))
        return res


###404左叶子之和
class Solution:
    def sum_left_leaves(self,root:TreeNode):
        """
        左叶子的判定，需要根据当前节点的父亲节点和当前节点共同确实。
        父亲节点的左节点不为空，当前节点左右孩子为空，则当前节点为左叶子节点
        后序遍历：先左后右再中
        递归遍历：三部曲 1.确定递归函数的参数和返回值。2.确定终止条件。3.确定单层逻辑

        """
        if root is None:
            return 0
        left_value = self.sum_left_leaves(root.left) ##左
        right_value = self.sum_left_leaves(root.right) ##右
        middle_value = 0 ##中

        if root.left and root.left.left is None and root.left.right is None:  ##左叶子判断
            middle_value = root.left.val
        res = middle_value+left_value+right_value
        return res

    def sum_left_leaves_itertor(self,root:TreeNode):
        """
        迭代法，使用栈模拟递归过程
        """
        stack = []
        if root is None:
            return 0 
        stack.append(root)
        res = 0
        while stack:
            node = stack.pop()
            if node.left and node.left.left is None and node.left.right is None:
                res += node.left.val
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return res


###513找树左下角的值
class Solution:
    def __init__(self):
        self.max_len = -1
        self.max_left_value = 0
    def findBottomLeftValue(self, root: TreeNode) -> int:
        """
        思路:首先需要找到最后一层，然后找最左边的值
        层次遍历最简单，层次遍历只需要遍历到最后一层，然后返回最后一层的第一个元素即可，递归稍微复杂
        先使用递归来解决
        """
        self.traversal(root,0)
        return self.max_left_value
    def traversal(self,root:TreeNode,left_len:int):
        """递归三部曲，确定递归函数参数和返回值，返回值：如果遍历整棵树就不需要返回值，如果是遍历某一条固定路径，则需要返回值"""
        if root.left is None and root.right is None:  ###当遇到叶子节点时，需要统计最大深度
            if left_len > self.max_len:
                 self.max_len = left_len
                 self.max_left_value = root.val
            

        if root.left:
            left_len += 1
            self.traversal(root.left,left_len)
            left_len -= 1  ##回溯
        if root.right:
            left_len += 1
            self.traversal(root.right,left_len)
            left_len -= 1
        return


class Solution:
    def findBottomLeftValue(self,root:TreeNode):
        """层次遍历，返回最后一层的第一个元素"""
        from collections import deque

        if root is None:
            return 0
        res = 0
        deq = deque()
        deq.append(root)
        while deq:
            size = len(deq)
            for i in range(size):
                node = deq.popleft()
                if i == 0:  ###拿到当前层第一个元素
                    res = node.val
                if node.left:
                    deq.append(node.left)
                if node.right:
                    deq.append(node.right)
        return res

class Solution:
    def levelOrder(self, root: TreeNode):
        """二叉树层序遍历

        Args:
            root (TreeNode): [description]

        Returns:
            List[List[int]]: [description]
        """
        from collections import deque

        dq = deque()
        result = []
        if root is None:
            return []
        dq.append(root)
        while dq:
            res = []
            size = len(dq)
            for i in range(size):
                node = dq.popleft()
                res.append(node.val)
                if node.left:
                    dq.append(node.left)
                if node.right:
                    dq.append(node.right)
            result.append(res)
        return result


if __name__ == "__main__":
    pass
    
