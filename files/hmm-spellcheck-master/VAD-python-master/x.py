input_ = "iki saat on derece ayarla"

def get_exact_clock(input_):
    try:
        input_ = clean_numbers(input_)
        double_digits = re.findall(
            r'(\bon \b((bir)|(iki)|(üç)|(dört)|(beş)|(altı)|(yedi)|(sekiz)|(dokuz))([ye|ya|a|e])*\b)',
            input_)
        if (len(double_digits) > 0):
            input_ = translate_tens(input_, double_digits)

        new_sentence = ""
        translated_tens_list = list(special_tens_turk_to_eng.values())
        for token in input_.split():
            if token in nums_turk_to_eng:
                new_sentence += nums_turk_to_eng[token] + " "
            elif (token in translated_tens_list):
                new_sentence += token + " "
        time_string = t2d.convert(new_sentence).strip()

        # examples:
        # beş, on, on beş (only 1-19)
        if (len(new_sentence.split()) == 1):
            first_part = time_string
            second_part = '00'
        else:
            # four digit input
            if (len(time_string) == 4):
                first_part = time_string[:2]
                second_part = time_string[2:]
            else:
                # add a special seperator ;;;
                new_sentence = mark_last_single(new_sentence)
                time_string = t2d.convert(new_sentence).strip()
                splitted = time_string.split(';;;')
                first_part = splitted[0]
                second_part = splitted[1]


        if second_part is None or second_part == "":
            second_part = '00'

        if len(first_part)==1:
            first_part = '0' + first_part

        if len(second_part) == 1:
            second_part = '0' + second_part

        return f"{first_part}:{second_part}"

    except:
        return -1

clock = get_exact_clock(input_)
if 'dakika' in input_ or 'saatlık' in input_ or 'saatlik' in input_ or 'saate' in input_:
    hour = int(clock.split(':')[0])
    mins = int(clock.split(':')[1])

    print(hour)