# -*- coding: utf-8 -*-

import datetime
# from qq import sendMail
# from twilio.rest import Client
# from LunarSolarConverter import LunarSolarConverter, Solar, Lunar


def load_file(filename,solar_day,solar_month):
    with open(filename,encoding='utf8') as read_file:
        name_list,office_list,position_list = [],[],[]
        for line in read_file:
            employee_info_list = line.strip().split()
            name = employee_info_list[0].strip()
            birth_month = int(employee_info_list[1].strip())
            birth_day = int(employee_info_list[2].strip())
            office = employee_info_list[3].strip()
            if len(employee_info_list) == 5:
                position_type = employee_info_list[4]
            else:
                position_type = ' '

            if birth_month == solar_month and birth_day == solar_day:
                name_list.append(name)
                office_list.append(office)
                position_list.append(position_type)
            
    return name_list,office_list,position_list
        




def main():
    now = datetime.datetime.now()
    print('公历-----------------------------------')
    print('日期时间: ', now)

    solar_year = now.year
    solar_month = now.month
    solar_day = now.day-2
    print('年: ', solar_year)
    print('月: ', solar_month)
    print('日: ', solar_day)
    print('\n')
    
    birthday = str(solar_month)+'月'+str(solar_day)+'日'
    # print('阴历-----------------------------------')
    # converter = LunarSolarConverter()
    # solar = Solar(solar_year, solar_month, solar_day)
    # lunar = converter.SolarToLunar(solar)
    jiujiang_name_list,jiujiang_office_list,jiujiang_position_list = load_file('jiujiang_birthday.txt',solar_day=solar_day,solar_month=solar_month)
    nanchang_name_list,nanchang_office_list,nanchang_position_list = load_file('nanchang_birthday.txt',solar_day=solar_day,solar_month=solar_month)



    jiujiang_info_list =[]
    for jiujiang_name,jiujiang_office,jiujiang_position in zip(jiujiang_name_list,jiujiang_office_list,jiujiang_position_list):
        if jiujiang_position.strip():
            employee_info = str(jiujiang_office)+str(jiujiang_position)+str(jiujiang_name) 
            jiujiang_info_list.append(employee_info)
        else:
            employee_info = str(jiujiang_office)+str(jiujiang_name) 
            jiujiang_info_list.append(employee_info)

    nanchang_info_list =[]
    for nanchang_name,nanchang_office,nanchang_position in zip(nanchang_name_list,nanchang_office_list,nanchang_position_list):
        if nanchang_position.strip():
            employee_info = str(nanchang_office)+str(nanchang_position)+str(nanchang_name) 
            nanchang_info_list.append(employee_info)
        else:
            employee_info = str(nanchang_office)+str(nanchang_name) 
            nanchang_info_list.append(employee_info)

    jiujiang_msg = ' '.join(jiujiang_info_list)  
    nanchang_msg = ' '.join(nanchang_info_list)
    title = '九江银行生日提醒'
    
    msg = f'领导好， 明天过生日的有:九江地区{jiujiang_msg},南昌地区{nanchang_msg},具体日期是{birthday}'
    # sendMail(title, msg)
    print(msg)


if __name__ == "__main__":
    pass