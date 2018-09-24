# Definition for singly-linked list:

import copy

class ListNode(object):
    def __init__(self, x):
        self.value = x
        self.next = None


def isListPalindrome(l):
    head = ListNode(None)
    inv_list = ListNode(None)
    head.next = l
    fast = head
    slow = head
    if fast:
        while fast.next:
            if fast.next.next:
                print('fast', fast.next.next.value)
                fast = fast.next.next
                slow = slow.next
            else:
                fast = fast.next
                slow = slow.next

        print('now make inv link')
        curr = ListNode(None)
        while slow.next:
            slow = slow.next
            print('slow.value', slow.value)

            inv_list.next = copy.copy(slow)
            if curr:
                inv_list.next.next = curr
            curr = inv_list.next

    pos = head
    neg = inv_list
    while neg.next:
        if neg.next.value:
            pos = pos.next
            neg = neg.next
            print(pos.value, neg.value)
        else:
            break


if __name__ == '__main__':
    l = ListNode(1)
    l.next = (ListNode(2))
    l.next.next = (ListNode(2))
    l.next.next.next = (ListNode(3))
    l.next.next.next.next = (ListNode(3))
    print(isListPalindrome(l))
