class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def sortedArrayToBST(self, nums):
        if nums is None:
            return None
        begin = 0
        end = len(nums) - 1
        return self.buildTree(nums, begin, end)

    def buildTree(self, nums: list, begin: int, end: int):
        if begin > end:
            return None
        mid = (begin + end) >> 1
        root = TreeNode(nums[mid])

        root.left = self.buildTree(nums, begin, mid - 1)
        root.right = self.buildTree(nums, mid + 1, end)
        return root


def breadh_travel(root):
    """广度优先遍历"""
    if root is None:
        return
    queue = []
    queue.append(root)
    while len(queue) > 0:
        node = queue.pop(0)
        try:
            print(node.val, end=' ')
            if node.left or node.right:
                if node.left:
                    queue.append(node.left)
                else:
                    queue.append('null')
                if node.right:
                    queue.append(node.right)
                else:
                    queue.append('null')
        except:
            print(node, end=' ')


ret = Solution().sortedArrayToBST([-10,-3,0,5,9])
breadh_travel(ret)
