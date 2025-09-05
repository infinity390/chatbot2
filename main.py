import itertools
import copy
class TreeNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []
    def __repr__(self):
        return string_equation(self)
    def fx(self, name):
        return TreeNode(name, [self])
    def __hash__(self):
        return hash(str_form(self))
    def __eq__(self, other):
        return str_form(self) == str_form(other)
def tree_form(tabbed_strings):
    lines = tabbed_strings.split("\n")
    root = TreeNode("Root") # add a dummy node
    current_level_nodes = {0: root}
    stack = [root]
    for line in lines:
        level = line.count(' ') # count the spaces, which is crucial information in a string representation
        node_name = line.strip() # remove spaces, when putting it in the tree form
        node = TreeNode(node_name)
        while len(stack) > level + 1:
            stack.pop()
        parent_node = stack[-1]
        parent_node.children.append(node)
        current_level_nodes[level] = node
        stack.append(node)
    return root.children[0] # remove dummy node

# convert tree into string representation
def str_form(node):
    def recursive_str(node, depth=0):
        result = "{}{}".format(' ' * depth, node.name) # spacings
        for child in node.children:
            result += "\n" + recursive_str(child, depth + 1) # one node in one line
        return result
    return recursive_str(node)

def string_equation(equation_tree):
    if not equation_tree.children:
        # Leaf nodes
        if equation_tree.name.startswith("v_"):
            return equation_tree.name[2:].upper()  # v_x → X
        elif equation_tree.name.startswith("d_"):
            return equation_tree.name[2:]  # d_5 → 5
        else:
            return equation_tree.name

    # Operator mapping for infix style
    op_map = {
        "f_dot": " . ",
        "f_add": " + ",
        "f_mul": " * ",
        "f_pow": " ^ "
    }

    if equation_tree.name in op_map:
        # Infix operator
        op = op_map[equation_tree.name]
        parts = [string_equation(child) for child in equation_tree.children]
        return "(" + op.join(parts) + ")"

    elif equation_tree.name == "f_len":
        # Length style with bars
        inner = string_equation(equation_tree.children[0])
        return f"|{inner}|"

    elif equation_tree.name.startswith("f_"):
        # Function style
        func_name = equation_tree.name[2:]  # Remove "f_"
        args = ", ".join(string_equation(child) for child in equation_tree.children)
        return f"{func_name}({args})"

    else:
        # Unknown node style
        args = ", ".join(string_equation(child) for child in equation_tree.children)
        return f"{equation_tree.name}({args})"


class word:
    def __init__(self, param):
        if isinstance(param, str):
            self.exp = [[param]]
        else:
            self.exp = [[item2 for item2 in item if item2 != "@"] for item in param]

    def __mul__(self, other):
        if isinstance(self, int):
            p = other
            for i in range(self-1):
                p += other
            return p
        elif isinstance(other, int):
            p = self
            for i in range(other-1):
                p += self
            return p
        output = []
        for item in itertools.product(self.exp, other.exp):
            element = []
            for item2 in item:
                element += item2
            output.append(element)
            
        return word(output)
    def __add__(self, other):
        return word(self.exp+other.exp)
    def __and__(self, other):
        output = []
        for item in zip(self.exp, other.exp):
            element = []
            for item2 in item:
                element += item2
            output.append(element)
        return word(output)
    
    def __repr__(self):
        return "\n".join([" ".join(item) for item in self.exp])
def summation(string):
    string = string.split(" ")
    p = word(string[0])
    for item in string[1:]:
        p = p + word(item)
    return p
def product(string):
    string = string.split(" ")
    p = word(string[0])
    for item in string[1:]:
        p = p * word(item)
    return p

s = " ".join([item.replace(" ", "-") for item in str(summation("lucy's mary's john's bob's my your her his")*summation("father mother son daughter")).split("\n")])
actor = summation("you i john bob lucy mary she he "+s)
tragedy = product("car accident")
todie = summation("die perish")
adjective = summation("not-dead not-sad dead alive not-alive happy sad not-happy")
action = summation("die walk run kill")
gender = summation("girl boy")
noun = summation("girl boy criminal innocent")
thing = summation("book watch")
relation = summation("friend")
const = {}
const["noun"] = noun
const["actor"] = actor
const["action"] = action
const["todie"] = todie
const["tragedy"] = tragedy
const["adjective"] = adjective
const["thing"] = thing
def compute(eq):
    vn = 0
    def compute2(eq):
        global const
        nonlocal vn
        if eq.name[:2] == "v_":
            w = copy.deepcopy(const[eq.name[2:]])
            for i in range(len(w.exp)):
                for j in range(len(w.exp[i])):
                    w.exp[i][j] = w.exp[i][j]+"_"+str(vn)
            vn += 1
            return w
        elif eq.name[:2] == "d_":
            return word(eq.name[2:])
        elif eq.name[:2] == "f_":
            w = None
            if eq.name == "f_add":
                w = compute2(eq.children[0])
                for child in eq.children[1:]:
                    w += compute2(child)
                
            elif eq.name == "f_repeat":
                w = compute2(eq.children[0])
                if isinstance(w, int):
                    out = compute2(eq.children[1])
                    out2 = word("@")
                    out2.exp = []
                    for item in out.exp:
                        for i in range(w):
                            out2.exp.append(item)
                    w = out2
            elif eq.name == "f_mul":
                w = compute2(eq.children[0])
                for child in eq.children[1:]:
                    tmp = compute2(child)
                    w *= tmp
            elif eq.name == "f_dot":
                w = compute2(eq.children[0])
                w2 = compute2(eq.children[1])
                for i in range(len(w.exp)):
                    w.exp[i] = w.exp[i] + w2.exp[i]
            elif eq.name == "f_pow":
                w = compute2(eq.children[0])
                out = w
                n = compute2(eq.children[1])
                for i in range(n-1):
                    out *= w
                w = out
            elif eq.name == "f_len":
                w = compute2(eq.children[0])
                return len(w.exp)
            elif eq.name == "f_article":
                w = compute2(eq.children[0])
                for i in range(len(w.exp)):
                    if w.exp[i][0][0] in "aeiou":
                        w.exp[i] = ["an"]
                    else:
                        w.exp[i] = ["a"]
            elif eq.name == "f_aux":
                w = compute2(eq.children[0])
                for i in range(len(w.exp)):
                    if len(w.exp[i]) == 1 or w.exp[i][1] == "is":
                        d = None
                        if "_" in w.exp[i][0]:
                            d = w.exp[i][0][-1:]
                            w.exp[i][0] = w.exp[i][0][:-2]
                        if w.exp[i][0] in ["i", "you", "we", "they"]:
                            if d is None:
                                w3 = ""
                            else:
                                w3 = "_" + d
                            w.exp[i] = [{"i":"am", "you":"are", "we":"are", "they":"are"}[w.exp[i][0]] +w3]
                        else:
                            w.exp[i] = ["is"]
                    elif w.exp[i][1] == "have":
                        d = None
                        if "_" in w.exp[i][0]:
                            d = w.exp[i][0][-1:]
                            w.exp[i][0] = w.exp[i][0][:-2]
                        if w.exp[i][0] in ["i", "you", "we", "they"]:
                            w.exp[i] = [{"i":"have", "you":"have", "we":"have", "they":"have"}[w.exp[i][0]] + "_" + d]
                        else:
                            w.exp[i] = ["has"]
            elif eq.name == "f_cont":
                pass
            elif eq.name == "f_obj":
                w = compute2(eq.children[0])
                
                for i in range(len(w.exp)):
                    d = None
                    if "_" in w.exp[i][0]:
                        d = w.exp[i][0][-1:]
                        w.exp[i][0] = w.exp[i][0][:-2]
                    w3 = None
                    if d is None:
                        w3 = ""
                    else:
                        w3 = "_"+d
                    if w.exp[i][0] in ["i", "they", "he", "she"]:
                        w.exp[i] = [{"i":"me", "he":"him", "she":"her", "they":"them"}[w.exp[i][0]] + w3]
                    else:
                        w.exp[i] = [w.exp[i][0] +w3]
            elif eq.name == "f_verb":
                w = compute2(eq.children[0])
                
                for i in range(len(w.exp)):
                    
                    d = None 
                    for j in range(len(w.exp[i])):
                        if "_" in w.exp[i][j]:
                            
                            tmp = w.exp[i][j][-1:]
                            w.exp[i][j] = w.exp[i][j][:-2]
                            if tmp.isdigit():
                                d = tmp
                    w3 = None
                    if d is None:
                        w3 = ""
                    else:
                        w3 = "_"+d
                    if w.exp[i][0] in ["i", "you", "we", "they"]:
                        w.exp[i] = [w.exp[i][-1]+w3]
                    else:
                        w2 = w.exp[i][-1]
                        if w2 == "have":
                            w2 = "has"
                        elif w2[-1] in ["e", "t", "l", "k", "n"]:
                            w2 += "s"
                        else:
                            w2 += "es"
                        w.exp[i] = [w2 + w3]
                    
            elif eq.name == "f_past":
                w = compute2(eq.children[0])
                
                for i in range(len(w.exp)):
                    d = None
                    if "_" in w.exp[i][0]:
                        d = w.exp[i][0][-1:]
                        w.exp[i][0] = w.exp[i][0][:-2]
                    w3 = None
                    if d is None:
                        w3 = ""
                    else:
                        w3 ="_"+d
                    dic = {"died":["die", "dies", "dying"], "was":["is", "am"], "were":["are"], "ran":["runs", "run", "running"]}
                    sel = []
                    for key in dic.keys():
                        sel += dic[key]
                    if w.exp[i][0] in sel:
                        for key in dic.keys():
                            if w.exp[i][0] in dic[key]:
                                
                                w.exp[i] = [key+w3]
                                break
                    else:
                        w2 = w.exp[i][0]
                        if w2[-1] == "s":
                            w2 = w2[:-1]
                        if w2[-1] in ["l", "k"]:
                            w.exp[i] = [w2+"ed"+w3]
                        else:
                            w.exp[i] = [w2+"d"+w3]
            return w
    s =compute2(eq)
    for i in range(len(s.exp)-1,-1,-1):
        for j in range(len(s.exp[i])-1,-1,-1):
            if s.exp[i][j] == "#":
                s.exp[i].pop(j)
        if s.exp[i] ==[]:
            s.exp.pop(i)
    return s
def rmdash(w):
    w = copy.deepcopy(w)
    for i in range(len(w.exp)):
        for j in range(len(w.exp[i])):
            if "_" in w.exp[i][j]:
                w.exp[i][j] = w.exp[i][j][:-2]
    return w

def pro(eq):
    male = None
    female = None
    def helper(eq):
        nonlocal female
        nonlocal male
        if eq.name in ["mary", "lucy"]:
            female = eq
        elif eq.name in ["john", "bob"]:
            male = eq
        if eq.name in ["she", "her"] and female is not None:
            return female
        elif eq.name in ["he", "him", "his"] and male is not None:
            return male
        return TreeNode(eq.name, [helper(child) for child in eq.children])
    return helper(helper(eq))
def matcheq(eq, sentence):
    out = compute(tree_form(eq))
    out2 = str(rmdash(out)).replace("-", " ").split("\n")
    index = None
    out3= None
    
    if sentence in out2:
        
        index = out2.index(sentence)
        
        out3 = str(out).split("\n")[index]
    
    if index is not None:
        
        dic = {}
        for item in out3.split(" "):
            if "_" not in item:
                continue
            a, b= item.split("_")
            if int(b) not in dic.keys():
                dic[int(b)] = a
            else:
                dic[int(b)] += " "+a
        
        return dic
    
    return None
def conv_word(w):
    w = w.replace("-", " ")
    
    opposite2 = [("happy", "sad"), ("boy","girl"), ("criminal","innocent"), ("dead","alive")]
    
    for item in opposite2:
        if w == "not " + item[0]:
            return simplify3(conv_word(item[0]))
        if w == "not " + item[1]:
            return conv_word(item[0])
    if " " in w:
        eq = conv_word(w.split(" ")[0])
        for item in w.split(" ")[1:]:
            eq = eq.fx(conv_word(item).name)
        return eq
    
    for item in opposite2:
        if w == item[1]:
            return simplify3(conv_word(item[0]))
        
    if w in ["you", "your"]:
        return tree_form("you")
    if w in ["i", "my", "me"]:
        return tree_form("i")
    if w[-2:] == "'s":
        return tree_form(w[:-2])
    
    if w[-2:] in ["ed", "es"]:
        if w[-3] in ["i"]:
            return tree_form(w[:-1])
        return tree_form(w[:-2])
    
    return tree_form(w)
def alltense(eq):
    def rmnode(eq, node):
        if eq.name == node:
            return eq.children[0]
        return TreeNode(eq.name, [rmnode(child, node) for child in eq.children])
    eq = tree_form(eq)
    out = [eq]
    out += [rmnode(eq, "f_past")]
    out += [rmnode(rmnode(eq, "f_cont"), "f_past")]
    out += [rmnode(eq, "f_cont")]
    return list(set(out))
def plural(eq):
    return False
def sort_logic(eq):
    if len(eq.children) == 2:
        if eq.children[1].name in ["livingstate", "emotionalstate"]:
            return TreeNode(eq.name, eq.children[::-1])
    return eq
def print_logic(eq, poss=False):
    if eq.name == "exchange" and eq.children[0].name == "poss":
        a = print_logic(eq.children[0].children[0])
        r = "take"
        if a in ["i", "you", "they", "we"]:
            pass
        else:
            r = "takes"
        return " ".join([a, r, "the", eq.children[1].name])
    if eq.name == "put" and eq.children[0].name == "action":
        a = print_logic(eq.children[0].children[0])
        w = eq.children[1].name
        if a in ["i", "you", "they", "we"]:
            pass
        else:
            w += "s"
        return " ".join([a, w])
    if eq.name == "put" and eq.children[0].name == "poss":
        a = print_logic(eq.children[0].children[0])
        r = "have"
        if a in ["i", "you", "they", "we"]:
            pass
        else:
            r = "has"
        return " ".join([a, r, "an" if eq.children[1].name[0] in ["a", "e", "i", "o", "u"] else "a", eq.children[1].name])
    if eq.name == "put" and eq.children[0].name == "want":
        a = print_logic(eq.children[0].children[0])
        r = "want"
        if a in ["i", "you", "they", "we"]:
            pass
        else:
            r += "s"
        return " ".join([a, r, "to", eq.children[1].name])
    if eq.name == "act":
        a = print_logic(eq.children[0])
        b = print_logic(eq.children[1])+"ed"
        c = print_logic(eq.children[2])
        if c == "i":
            c = "me"
        return " ".join([a, b, c])
    if eq.name in ["equal"]:
        a = print_logic(eq.children[0])
        aux = "are" if plural(eq.children[0]) else "is"
        if a == "i":
            aux = "am"
        if a == "you":
            aux = "are"
        b = print_logic(eq.children[1])
        if b == "i":
            b = "me"
        return " ".join([a, aux, b])
    if eq.name == "not":
        return " ".join(["not", print_logic(eq.children[0])])

    
    if eq.children is None or len(eq.children) == 0:
        if poss:
            dic = {"you": "your", "i":"my"}
            if eq.name in dic.keys():
                
                return dic[eq.name]
            return eq.name + "'s"
        return eq.name
    
    if eq.name in ["livingstate", "emotionalstate"]:
        return print_logic(eq.children[0])
    if eq.name in ["daughter", "mother", "father", "son"]:
        return " ".join([print_logic(eq.children[0], poss=True), eq.name])
def simplify(eq):
    if (eq.name in ["mother", "father"] and eq.children and eq.children[0].name in ["son", "daughter"]):
        return simplify(eq.children[0].children[0])
    if eq.name == "not" and eq.children and eq.children[0].name == "not":
        return simplify(eq.children[0].children[0])
    
    return TreeNode(eq.name, [simplify(child) for child in eq.children])
def simplify2(eq):
    while True:
        orig = str_form(eq)
        eq = simplify(eq)
        if str_form(eq) == orig:
            return eq
def simplify3(eq):
    if eq.name == "alive":
        return tree_form("dead")
    if eq.name == "dead":
        return tree_form("alive")
    if eq.name == "happy":
        return tree_form("sad")
    if eq.name == "sad":
        return tree_form("happy")
    if eq.name == "girl":
        return tree_form("boy")
    if eq.name == "boy":
        return tree_form("girl")
    if eq.name == "criminal":
        return tree_form("innocent")
    if eq.name == "innocent":
        return tree_form("criminal")
    return TreeNode(eq.name, [simplify3(child) for child in eq.children])
def simplify4(eq):
    if eq.name == "gender" and eq.children[0].name in ["father", "son", "john", "bob"]:
        return tree_form("boy")
    if eq.name == "gender" and eq.children[0].name in ["daughter", "mother", "lucy"]:
        return tree_form("girl")
    return TreeNode(eq.name, [simplify4(child) for child in eq.children])
def add_gender(actor):
    if actor.name in ["father", "son", "john", "bob"]:
        return [TreeNode("equal", [tree_form("boy"), actor.fx("gender")])]
    if actor.name in ["mother", "daughter", "mary", "lucy"]:
        return [TreeNode("equal", [tree_form("girl"), actor.fx("gender")])]
    return []
class Logic:
    def __init__(self, code, depth=0):
        self.depth = depth
        self.code = copy.deepcopy(code)
        other = []
        def find_gender(eq):
            out = add_gender(eq)
            for child in eq.children:
                out += find_gender(child)
            return out
        for item in self.code:
            other += find_gender(item)
            if item.name == "act":
                
                dic2 = {"kill":"dead"}
                if item.children[1].name in dic2.keys():
                    
                    other += [TreeNode("equal", [item.children[2].fx("livingstate"), tree_form(dic2[item.children[1].name])])]
                    other += [TreeNode("equal", [item.children[0].fx("guiltystate"), tree_form("criminal")])]
        self.code += list(set(other))
        self.cat = []
        hist = []
        
        for i in range(len(self.code)-1,-1,-1):
            if self.code[i].name in ["put", "exchange"]:
                
                if self.code[i].children[1] in hist:
                    self.code.pop(i)
                elif self.code[i].name == "exchange":
                    hist.append(self.code[i].children[1])
                    self.code[i].name = "put"

        dic = {}
        for i in range(len(self.code)-1,-1,-1):
            if self.code[i].name == "put":
                if str_form(self.code[i].children[0]) not in dic.keys():
                    dic[str_form(self.code[i].children[0])] = [self.code[i].children[1]]
                else:
                    dic[str_form(self.code[i].children[0])] += [self.code[i].children[1]]
                self.code.pop(i)
                
        for key in dic.keys():
            self.code.append(TreeNode("equal", [tree_form(key), TreeNode("list", list(sorted(dic[key], key=lambda x: str_form(x))))]))
        

        
        for item in self.code:
            if item.name in ["equal"]:
                self.cat.append([str_form(x) for x in item.children])
            else:
                self.cat.append([str_form(item)])
        
        self.merge()
        self.merge3()
        self.cat = [list(set([str_form(simplify2(tree_form(y))) for y in x])) for x in self.cat]
        self.code, self.cat = self.fix_contradict()
        
    def __repr__(self):
        s = ""
        for item in self.cat:
            s+="=========\n"
            for item2 in item:
                s += str(tree_form(item2))+"\n"
        return s
    def fix_contradict(self):
        
        if self.depth > 0 or self.is_contradict():
            return self.code, self.cat
        orig = copy.deepcopy(self.code)
        for i in range(len(self.code)):
            
            orig.pop(i)
            logic = Logic(orig, self.depth+1)
            if logic.is_contradict():
                return logic.code, logic.cat
            orig.insert(i, self.code[i])
    def is_contradict(self):
        lst = [("happy","sad"),("dead","alive"),("boy","girl"),("criminal","innocent")]
        for item in self.cat:
            if any(x[0] in item and x[1] in item for x in lst):
                return False
        return True
    def merge(self):
        
        while True:
            none_left = True
            if len(self.cat) > 1:
                for item in itertools.combinations(range(len(self.cat)), 2):
                    lst = list(set(self.cat[item[0]]+self.cat[item[1]]))
                    if len(lst) < len(self.cat[item[0]])+len(self.cat[item[1]]):
                        item = sorted(list(item))
                        self.cat.pop(item[1])
                        self.cat.pop(item[0])
                        self.cat.append(lst)
                        none_left = False
                        break
            if none_left:
                break
            
    def merge3(self):
        def rep(question):
            out = []
            out2 = [rep(child) for child in question.children]
            for item in itertools.product(*out2):
                out.append(TreeNode(question.name, list(item)))
            tmp = self.query2(question)
            if tmp:
                out += [tree_form(x) for x in tmp]
            return out
        for i in range(len(self.cat)):
            out = []
            for j in range(len(self.cat[i])):
                q = tree_form(self.cat[i][j])
                out += [str_form(x) for x in rep(q)]
           
            self.cat[i] = list(set(self.cat[i] + out))
            
    def query2(self, question):
        for item in self.cat:
            if str_form(question) in item:
                return item
        return None
    def query(self, question):
        question = simplify2(question)
        helper = None
        def match2(matcher, data):
            
            nonlocal helper
            if matcher.name == "*":
                
                helper = str_form(data)
                return True
            if data.name != matcher.name:
                return False
            if tree_form("+") in matcher.children and set([child for child in matcher.children if child != tree_form("+")])<=set(data.children):
                return True
            
            if len(matcher.children) != len(data.children):
                return False
            
            
            
            return all(match2(matcher.children[i], data.children[i]) for i in range(len(matcher.children)))
        typeq = question.name
        
        out2 = []
        for i in range(len(self.cat)):
            if typeq == "?x":
                for item in self.cat[i]:
                    
                    if match2(question.children[0], tree_form(item)):
                        
                        out2.append(helper)
            elif typeq == "?":
                if question.children[0].name == "belong" and\
                   str_form(simplify4(question.children[0].children[0])) in self.cat[i] and\
                   str_form(simplify4(question.children[0].children[1])) in self.cat[i]:
                    return True
                if question.children[0].name == "belong" and\
                   str_form(simplify3(simplify4(question.children[0].children[0]))) in self.cat[i] and\
                   str_form(simplify4(question.children[0].children[1])) in self.cat[i]:
                    return False
            elif typeq == "?*" and (str_form(question.children[0]) in self.cat[i] or\
                                    any(match2(question.children[0], tree_form(child)) for child in self.cat[i])):
                out = []
                for item in self.cat[i]:
                    out.append(item)
                return out
        if typeq == "?":
            return None
        return out2
def fliplogic(eq):
    if eq.name == "you":
        return tree_form("i")
    if eq.name == "i":
        return tree_form("you")
    return TreeNode(eq.name, [fliplogic(child) for child in eq.children])
def jointree(eq, eq2):
    if eq.children is None or len(eq.children)==0:
        return TreeNode(eq.name, [eq2])
    return TreeNode(eq.name, [jointree(child, eq2) for child in eq.children])
def wrap(adjective, actor):
    if adjective.name in ["criminal", "innocent"]:
        return actor.fx("guiltystate")
    if adjective.name in ["girl", "boy"]:
        return actor.fx("gender")
    if adjective.name in ["happy", "sad"]:
        return actor.fx("emotionalstate")
    elif adjective.name in ["dead", "alive"]:
        return actor.fx("livingstate")
eq = """f_mul
 f_dot
  f_repeat
   f_len
    v_action
   v_actor
  f_past
   f_verb
    f_mul
     v_actor
     v_action"""
'''
eq = tree_form(eq)
print(eq)
print(rmdash(compute(eq)))
'''
def convert2logic(sentence):
    f = """f_mul
 d_who
 f_add
  d_have
  d_has
 f_dot
  f_article
   v_thing
  v_thing"""
    for eq in alltense(f):
        
        eq = str_form(eq)
        dic = matcheq(eq, sentence)
        if dic is not None:
            return [TreeNode("list", [conv_word(dic[1]), tree_form("+")]).fx("?*")]
    f= """f_mul
 f_dot
  v_actor
  f_verb
   f_mul
    v_actor
    d_take
 f_dot
  f_repeat
   f_len
    v_thing
   d_the
  v_thing
 f_add
  d_#
  f_mul
   d_from
   v_actor"""
    
    for eq in alltense(f):
        
        eq = str_form(eq)
        dic = matcheq(eq, sentence)
        if dic is not None:
            return [TreeNode("exchange", [conv_word(dic[0]).fx("poss"), conv_word(dic[3])])]
    
    f= """f_mul
 f_dot
  v_actor
  f_verb
   f_mul
    v_actor
    d_have
 f_dot
  f_article
   v_thing
  v_thing"""
    for eq in alltense(f):
        
        eq = str_form(eq)
        dic = matcheq(eq, sentence)
        if dic is not None:
            return [TreeNode("put", [conv_word(dic[0]).fx("poss"), conv_word(dic[3])])]
        
    f = """f_mul
 d_who
 f_past
  f_verb
   f_mul
    d_he
    v_action
 f_obj
  v_actor"""
    for eq in alltense(f):
        
        eq = str_form(eq)
        dic = matcheq(eq, sentence)
        if dic is not None:
            return [TreeNode("act", [tree_form("*"), conv_word(dic[0]), conv_word(dic[1])]).fx("?x")]


    f = """f_mul
 f_dot
  v_actor
  f_verb
   f_mul
    v_actor
    d_want
 d_to
 v_action"""
    for eq in [f]:
        dic = matcheq(eq, sentence)
        if dic is not None:
            return [TreeNode("put", [conv_word(dic[0]).fx("want"), conv_word(dic[2])])]
    
    f = """f_mul
 f_dot
  v_actor
  f_past
   f_aux
    v_actor
 v_adjective"""
    for eq in alltense(f):
        eq = str_form(eq)
        dic = matcheq(eq, sentence)
        if dic is not None:
            return [TreeNode("equal", [conv_word(dic[2]), wrap(conv_word(dic[2]), conv_word(dic[0]))])]

    f = """f_mul
 f_dot
  f_past
   f_aux
    v_actor
  v_actor
 v_adjective"""
    for eq in alltense(f):
        eq = str_form(eq)
        dic = matcheq(eq, sentence)
        if dic is not None:
            return [TreeNode("belong", [conv_word(dic[2]), wrap(conv_word(dic[2]), conv_word(dic[1]))]).fx("?")]
    f = """f_mul
 f_dot
  v_actor
  f_past
   f_aux
    v_actor
 f_obj
  v_actor"""
    for eq in alltense(f):
        eq = str_form(eq)
        dic = matcheq(eq, sentence)
        
        if dic is not None:
            
            return [TreeNode("equal", [conv_word(dic[2]), conv_word(dic[0])])]+ add_gender(conv_word(dic[0]))+ add_gender(conv_word(dic[2]))
    f = """f_mul
 d_who
 f_dot
  f_aux
   v_actor
  v_actor"""
    for eq in alltense(f):
        eq = str_form(eq)
        dic = matcheq(eq, sentence)
        if dic is not None:
            return [conv_word(dic[1]).fx("?*")]
    f = """f_mul
 d_who
 f_add
  d_is
  d_are
 v_adjective"""
    for eq in alltense(f):
        eq = str_form(eq)
        dic = matcheq(eq, sentence)
        if dic is not None:
            return [conv_word(dic[0]).fx("?*")]
    f = """f_mul
 f_dot
  f_aux
   v_actor
  v_actor
 d_a
 v_noun"""
    for eq in alltense(f):
        eq = str_form(eq)
        dic = matcheq(eq, sentence)
        if dic is not None:
            return [TreeNode("belong", [conv_word(dic[2]), wrap(conv_word(dic[2]), conv_word(dic[1]))]).fx("?")]
    f = """f_mul
 v_actor
 f_past
  v_action
 f_obj
  v_actor"""
    
    for eq in alltense(f):
        eq = str_form(eq)
        dic = matcheq(eq, sentence)
        if dic is not None:
            return [TreeNode("act", [conv_word(dic[0]), conv_word(dic[1]), conv_word(dic[2])])]
    f = """f_mul
 f_dot
  f_repeat
   f_len
    v_action
   v_actor
  f_past
   f_verb
    f_mul
     v_actor
     v_action"""
    
    for eq in alltense(f):
        eq = str_form(eq)
        dic = matcheq(eq, sentence)
        if dic is not None:
            return [TreeNode("put", [conv_word(dic[1]).fx("action"), conv_word(dic[3])])]
    f = """f_mul
 d_who
 d_is
 f_dot
  f_article
   v_noun
  v_noun"""
    
    for eq in alltense(f):
        eq = str_form(eq)
        dic = matcheq(eq, sentence)
        if dic is not None:
            
            return [conv_word(dic[1]).fx("?*")]

code = []
while True:
    s = input("chat = ")
    if s=="logic":
        s = Logic(code)
        code = s.code
        print(s)
        continue
    l = convert2logic(s)
    
    ques = False
    for item in l:
        if item.name not in ["?", "?x", "?*"]:
            code.append(pro(fliplogic(item)))
        else:
            l = item
            ques = True
            break
    if ques:
        if l.name == "?":
            
            logic = Logic(code)
            code = logic.code
            l = fliplogic(copy.deepcopy(l))
            
            l2 = TreeNode("belong", [simplify3(l.children[0].children[0]), l.children[0].children[1]]).fx("?")

            if logic.query(l) == True:
                print("yes")
            elif logic.query(l2) == True:
                print("no")
            else:
                print("i don't know")
        else:
            
            logic = Logic(code)
            code = logic.code
            l = fliplogic(l)
            noout = True
            for item in logic.query(l):
                
                item = tree_form(item)
                
                if (item.children is None or len(item.children)==0) and item.name in str(noun).split("\n"):
                    continue
                if item.name in ["list"]:
                    continue

                if not (item.children is None or len(item.children)==0) and\
                   (item.name in ["gender"] or "state" in item.name):
                    
                    item= item.children[0]
                
                item = sort_logic(item)
                item = print_logic(item)
                noout =False
                print(item)
            if noout:
                print("i don't know")
    else:
        l = l[0]
        item = pro(l)
        item = fliplogic(item)
        item = sort_logic(item)
        item = print_logic(item)
        print(item)
    print()
