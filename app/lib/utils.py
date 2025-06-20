def is_not_blank(s: str) -> bool:
  """
  检查字符串是否不是None且不是空字符串（包括只包含空白字符的字符串）

  参数：
  s (str): 要检查的字符串

  返回：
  bool: 如果字符串不是None且不是空字符串则返回True，否则返回False
  """
  return bool(s and s.strip())


def is_blank(s: str) -> bool:
  """
  检查字符串是否是None或空字符串（包括只包含空白字符的字符串）

  参数：
  s (str): 要检查的字符串

  返回：
  bool: 如果字符串是None或空字符串则返回True，否则返回False
  """
  return not is_not_blank(s)
