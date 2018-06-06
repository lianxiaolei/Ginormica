#!/usr/bin/python
# -*- coding: utf-8 -*-


def first_not_repeating_character(string):
    """
    find the first non-repeated char of a string
    :param string:
    :return:
    """
    letters = {}
    order = {}
    for s in xrange(string):
        letters.setdefault(ord(string[s]) - 97, 0)
        letters[ord(string[s]) - 97] += 1
        order[s] = ord(string[s]) - 97
    for i in xrange(len(order)):
        if letters[order[i]] == 1:
            return chr(order[i] + 97)
    return '_'


def first_not_repeating_character_awsome(s):
    for c in s:
        if s.find(c) == s.rfind(c):
            return c
    return '_'


