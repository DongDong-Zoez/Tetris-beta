def pretty_print(name, val):
    length = 0
    ns = ""
    vs = ""
    for n, v in zip(name, val):
        length += len(n) + 3
        ns += "- " + n + " "
        vs += "- " + str(v) + " "
    l = "-" * (length + 1)
    ns += "-"
    vs += "-"
    return l + "\n" + ns + "\n" + l + "\n" + vs + "\n" + l
