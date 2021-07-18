import pandas as pd

def read_file(filename):
    lines = []

    with open(filename) as f:
        lines = f.readlines()

    return lines


def process_rule(rule):
    for caracter in ['<', '>', '(', ')', '[', ']', '\n']:
        rule = rule.replace(caracter, '')

    rule = rule.split(' ')

    rule = [i for i in rule if i != '']

    del rule[0]

    output = ''
    for part in rule:
        if part.isnumeric():
            output += '|' + part
        elif part == '==':
            output += '|'
        elif ':' in part:
            output += '|' + part
        else:
            output += part + ' '

    rule = output.split('|')

    return {
        'rule': rule[0].strip(), #.split(' '),
        'implies': rule[2].strip(), #.split(' '),
        'coverage': '{:.2f}'.format(int(rule[1]) / 3184),
        'confidence': rule[4].replace('conf:', ''),
        'lift': rule[5].replace('lift:', ''),
    }


def filter(rules):
    df_rules = pd.DataFrame()
    for rule in rules:
        if '==> PERMANENCIA_PROLONGADA' in rule:
            rule = process_rule(rule)

            if ' ' not in rule['implies']:
                df_rules = df_rules.append(rule, ignore_index=True)

    return df_rules


def main():
    rules = read_file('storage/rules_apriori.txt')
    df_rules = filter(rules)


if __name__ == '__main__':
    main()
