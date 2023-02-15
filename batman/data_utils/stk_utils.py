import os
from shutil import copyfile
from random import random, randrange, sample
import datetime

INPUT_DIR = os.path.join(os.getcwd(), "inputs")
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")


def copy_ships(scenario_path='./MiamiPortScenario_copy/', num_ships=20, loc_mutation_factor=0.00000001,
               speed_mutation_factor=0.1, new_scen_path=''):
    for i in range(1, num_ships + 1):
        ship_path = os.path.join(scenario_path, f"Ship{i:02d}.sh")
        if len(new_scen_path) > 0:
            new_ship_path = os.path.join(new_scen_path, f"Ship{i:02d}.sh")
        else:
            new_ship_path = os.path.join(scenario_path, f"Ship{i:02d}_copy.sh")

        with open(new_ship_path, 'w') as new_f:
            with open(ship_path, 'r') as old_f:
                in_waypoints = False
                for line in old_f:
                    new_line = line
                    if 'BEGIN Waypoints' in line:
                        in_waypoints = True
                    elif 'END Waypoints' in line:
                        in_waypoints = False
                    elif in_waypoints and len(line.strip()) > 0:
                        waypoints = [float(x) for x in line.split()]
                        waypoints = [x + x * loc_mutation_factor *
                                     random() * randrange(-1, 2) for x in waypoints]
                        waypoints[4] += waypoints[4] * speed_mutation_factor * random() * randrange(-1, 2)
                        new_line = ' '.join([str(x) for x in waypoints])
                        new_line = '\t\t\t\t ' + new_line + '\n'

                    new_f.write(new_line)

        if len(new_scen_path) == 0:
            copyfile(new_ship_path, ship_path)
            os.remove(new_ship_path)


def set_ship_property(ship_id, property_name, new_property_value, property_is_3d=False,
                      scenario_path='./MiamiPortScenario_copy/'):
    ship_path = os.path.join(scenario_path, f"Ship{ship_id:02d}.sh")
    new_ship_path = os.path.join(scenario_path, f"Ship{ship_id:02d}_copy.sh")
    if property_is_3d:
        ship_path = ship_path + '3'

    with open(new_ship_path, 'w') as new_f:
        with open(ship_path, 'r') as old_f:
            for line in old_f:
                new_line = line
                if property_name in line:
                    new_line = property_name + '\t\t ' + str(new_property_value) + '\n'
                    if not property_is_3d:
                        new_line = '\t\t' + new_line
                new_f.write(new_line)
    copyfile(new_ship_path, ship_path)
    os.remove(new_ship_path)


def set_property_all_ships(property_name, new_property_value, property_is_3d=False, num_ships=20,
                           scenario_path='./MiamiPortScenario_copy/'):
    for i in range(1, num_ships + 1):
        ship_path = os.path.join(scenario_path, f"Ship{i:02d}.sh")
        new_ship_path = os.path.join(scenario_path, f"Ship{i:02d}_copy.sh")
        if property_is_3d:
            ship_path = ship_path + '3'

        with open(new_ship_path, 'w') as new_f:
            with open(ship_path, 'r') as old_f:
                for line in old_f:
                    new_line = line
                    if property_name in line:
                        new_line = property_name + '\t\t ' + str(new_property_value) + '\n'
                        if not property_is_3d:
                            new_line = '\t\t' + new_line
                    new_f.write(new_line)
        copyfile(new_ship_path, ship_path)
        os.remove(new_ship_path)


def get_vessel_type(ship_num):
    vessel_type_dict = {1: '37', 2: '35', 3: '35', 4: '35', 5: '30', 6: '35',
                        7: '35', 8: '35', 9: '35', 10: '30', 11: '60', 12: '35', 13: '35',
                        14: '35', 15: '35', 16: '35', 17: '35', 18: '35', 19: '37', 20: '35', 21: '35'}
    return vessel_type_dict[ship_num]


def format_reports_to_ais(output_path, reports_path):
    messages_list = []
    headers = 'MMSI,BaseDateTime,LAT,LON,SOG,Heading,VesselName,IMO,VesselType,Status,Behavior,Intent,'
    headers += 'BehaviorDescription\n'
    with open(reports_path, 'r') as reports_file:
        reports_file.readline()
        i = 1
        status = 'underway'
        behavior = 'normal'
        intent = 'benign'
        behavior_description = ''
        for line in reports_file:
            line = line.strip('\n')
            if 'Time (UTCG)' in line:
                i += 1
            elif len(line) > 0:
                ship_id = f"{i:09d}"
                vessel_name = f"Ship{i:02d}"
                imo = 'IMO' + ship_id
                vessel_type = get_vessel_type(i)
                message = line.split(',')
                message[0] = message[0][:-4]
                message.insert(0, ship_id)
                message.append(vessel_name)
                message.append(imo)
                message.append(vessel_type)
                message.append(status)
                message.append(behavior)
                message.append(intent)
                message.append(behavior_description)
                messages_list.append(message)

    messages_list = sorted(messages_list, key=lambda x: x[1])
    with open(output_path, 'w') as f:
        f.write(headers)
        for message in messages_list:
            f.write(','.join(message) + '\n')


def randomly_remove_ais_messages(ais_file, percentage_to_remove=0.05, new_file=''):
    with open(ais_file, 'r') as f:
        headers = f.readline()
        lines = f.readlines()
    messages_to_remove = sample(range(0, len(lines)), int(len(lines) * percentage_to_remove))
    for i in range(1, len(lines) - 1):
        if i in messages_to_remove:
            if len(lines[i + 1]) > 0 and 'normal' in lines[i + 1]:
                lines[i + 1] = lines[i + 1].replace('normal', 'unexpected AIS activity')
                lines[i + 1] = lines[i + 1][:-1] + 'ship maintenance,\n'
            if len(lines[i - 1]) > 0 and 'normal' in lines[i - 1]:
                lines[i - 1] = lines[i - 1].replace('normal', 'unexpected AIS activity')
                lines[i - 1] = lines[i - 1][:-1] + 'ship maintenance,\n'
            lines[i] = ''

    if len(new_file) > 0:
        with open(new_file, 'w') as f:
            f.write(headers)
            for line in lines:
                f.write(line)
    else:
        with open(ais_file, 'w') as f:
            f.write(headers)
            for line in lines:
                f.write(line)


def sort_ais_messages(ais_file, sort_by_col='BaseDateTime'):
    # for sorting by date, if necessary
    lines_dict = {}
    with open(ais_file, 'r') as f:
        headers = f.readline()
        headers_list = headers.split(',')
        col_index = headers_list.index(sort_by_col)

        for line in f:
            d = line.split(',')
            if len(d) > 0:
                d = d[col_index]
                if sort_by_col == 'BaseDateTime':
                    d = datetime.datetime.strptime(d, '%d %b %Y %X')
                lines_dict[line] = d

    sorted_data = {k: v for k, v in sorted(lines_dict.items(), key=lambda item: item[1])}

    with open(ais_file, 'w') as f:
        f.write(headers)
        for line in sorted_data.keys():
            f.write(line)


def process_all_days_ais_data(ais_file_format='./MiamiPort_AIS_data/MiamiPortScenario_day{:02d}.csv',
                              reports_file_format='./MiamiPortScenario_copy/reports/day{:02d}_all_ships.csv',
                              num_days=30, num_ships=20):
    for i in range(1, num_days + 1):
        format_reports_to_ais(ais_file_format.format(i), reports_file_format.format(i))
        randomly_remove_ais_messages(ais_file_format.format(i))
        sort_ais_messages(ais_file_format.format(i))


def process_single_day_ais_data(day,
                                ais_file_format='./MiamiPort_AIS_data/MiamiPortScenario_day{:02d}.csv',
                                reports_file_format='./MiamiPortScenario_copy/reports/day{:02d}_all_ships.csv',
                                num_ships=20):
    format_reports_to_ais(ais_file_format.format(day), reports_file_format.format(day))
    randomly_remove_ais_messages(ais_file_format.format(day))
    sort_ais_messages(ais_file_format.format(day))


def combine_ais_files(individual_files_folder='./MiamiPort_AIS_data/MiamiPortScenario_day{:02d}.csv',
                      output_file_path='./MiamiPort_AIS_data/MiamiPortScenario_all30days.csv',
                      num_days=30):
    headers = ''
    ais_rows = []
    for day in range(1, num_days + 1):
        with open(individual_files_folder.format(day), 'r') as f:
            headers = f.readline()
            for line in f:
                line = line.replace('5 Nov 2021', '{} Nov 2021'.format(day))
                ais_rows.append(line)

    with open(output_file_path, 'w') as f:
        f.write(headers)
        for line in ais_rows:
            f.write(line)


def convert_ais_to_sh_files(day: int,
                            start_hour: int = 21,
                            month: int = 11,
                            year: int = 2017,
                            input_dir=INPUT_DIR,
                            output_dir=OUTPUT_DIR):
    """
    Takes an AIS file sorted by ship and reformats information to place lat/lon/sog waypoints in *.sh
    files for STK to read as ship objects. The *.sh files should already exist and be populated with a
    ship's information, but this method will change the Start and Stop time of the ship and change
    the waypoints of the ship. This method will not create a new *.sh file from scratch, so there
    should be an existing *.sh file for 1-n_ships. This method requires the AIS messages to be sorted
    by BaseDateTime then by MMSI.

    Parameters
    -----------
    day: int
        Day of month (1-31)
    start_hour: int
        Hour of the day (0-23)
    month: int
        Month of the year (1-12)
    year: int
        Year (YYYY)
    input_dir: string
        Directory of input file containing AIS messages, sorted by MMSI, to be used for waypoints in *.sh files
    output_dir: string
        Directory in which output will be places.
    """
    date = '{}_{:02d}_{:02}T{:02d}'.format(year, month, day, start_hour)
    ship_fname_schema = input_dir + '/MiamiPort_Scen_{}_{:02d}_{:02d}/Ship{:02d}.sh'
    input_file = input_dir + '/AIS_{}_MiamiPort/{}_20ships_sorted_datetime_mmsi.csv'.format(date[:7], date)
    ship_dict = {}
    ship_static_info_dict = {}
    video_meta_data_file = output_dir + '/ship_static_data/AIS_{}_ship_static_data.csv'.format(date[:10])
    with open(video_meta_data_file, 'w') as meta_f:
        meta_f.write('STKObjName,MMSI,ShipType,Length,Width, AspectRatio(l/w)\n')

    with open(input_file, 'r') as in_f:
        in_f.readline()  # skip the headers
        for line in in_f:
            ais_message = line.split(',')
            ship_mmsi = ais_message[0]
            basedatetime = datetime.strptime(ais_message[1], '%Y-%m-%dT%H:%M:%S')
            lat, lon = ais_message[2], ais_message[3]
            sog = ais_message[4]
            ship_type = ais_message[10]
            ship_length, ship_width = ais_message[12], ais_message[13]
            aspect_ratio = ''
            if len(ship_length) > 0 and len(ship_width) > 0 and float(ship_width) > 0.:
                aspect_ratio = str(float(ship_length) / float(ship_width))

            if ship_mmsi not in ship_dict.keys():
                ship_dict[ship_mmsi] = []
                ship_static_info_dict[ship_mmsi] = [ship_type, ship_length, ship_width, aspect_ratio]
            ship_dict[ship_mmsi].append([basedatetime, lat, lon, sog])

    start_time = datetime.strptime('{}-{}-{:02d}T{:02}:00:00'.format(year, month, day, start_hour), '%Y-%m-%dT%H:%M:%S')
    stop_time = datetime.strptime('{}-{}-{:02d}T{:02}:00:00'.format(year, month, day, start_hour + 1),
                                  '%Y-%m-%dT%H:%M:%S')
    time_delta = stop_time - start_time

    # Loop through each ship in ship_dict; each ship has a list of waypoints
    stk_date_format = '%d %b %Y %H:%M:%S'
    ship_num = 1
    for mmsi, waypoints in ship_dict.items():
        # Make sure ships have a waypoint at the beginning and end of scenario
        # if waypoints[0][0] > start_time:
        #     first_wp = waypoints[0].copy()
        #     first_wp[0] = start_time
        #     waypoints.insert(0, first_wp)
        # if waypoints[-1][0] < stop_time:
        #     last_wp = waypoints[-1].copy()
        #     last_wp[0] = stop_time
        #     waypoints.append(last_wp)

        with open(ship_fname_schema.format(year, month, day, ship_num), 'r') as ship_f:
            with open(ship_fname_schema.format(year, month, day, ship_num) + 'cp', 'w') as new_ship_f:
                in_interval = False
                in_waypoints = False
                for line in ship_f:
                    new_line = line
                    if 'BEGIN Interval' in line:
                        in_interval = True
                    elif 'END Interval' in line:
                        in_interval = False
                    elif 'NumberOfWaypoints' in line:
                        new_line = '\t\tNumberOfWaypoints\t\t' + str(len(waypoints)) + '\n'
                    elif in_interval and 'StartTime' in line:
                        new_line = '\t\t\tStartTime\t\t' + start_time.strftime(stk_date_format) + '\n'
                    elif in_interval and 'StopTime' in line:
                        new_line = '\t\t\tStopTime\t\t' + stop_time.strftime(stk_date_format) + '\n'
                    elif in_interval and 'Start' in line:
                        new_line = '\t\t\tStart\t\t' + start_time.strftime(stk_date_format) + '\n'
                    elif in_interval and 'Stop' in line:
                        new_line = '\t\t\tStop\t\t' + stop_time.strftime(stk_date_format) + '\n'
                    elif 'TimeOfFirstWaypoint' in line:
                        new_line = '\tTimeOfFirstWaypoint\t\t' + start_time.strftime(stk_date_format) + '\n'
                    elif 'BEGIN Waypoints' in line:
                        in_waypoints = True
                    elif 'END Waypoints' in line:
                        while len(waypoints) > 0:
                            wp = waypoints.pop(0)
                            cur_time = wp[0]
                            time_percentage = str(1000 * (cur_time - start_time) / time_delta)
                            if float(wp[3]) > 0.0:
                                sog = str(float(wp[3]) * 0.514444)  # get SOG from knots to m/sec
                            else:
                                sog = '0.000001'
                            wp_line = ' '.join([time_percentage, wp[1], wp[2], '0.0', sog, '0.0'])
                            wp_line = '\t\t' + wp_line + '\n'
                            new_ship_f.write(wp_line)

                        in_waypoints = False
                    elif in_waypoints:  # skip old waypoints
                        new_line = ''

                    new_ship_f.write(new_line)
        print('Writing {} to Ship{:02d}.sh'.format(mmsi, ship_num))
        with open(video_meta_data_file, 'a') as meta_f:
            new_line = 'Ship{},{},'.format(ship_num, mmsi)
            new_line += ','.join(ship_static_info_dict[mmsi])
            meta_f.write(new_line + '\n')
        copyfile(ship_fname_schema.format(year, month, day, ship_num) + 'cp',
                 ship_fname_schema.format(year, month, day, ship_num))
        os.remove(ship_fname_schema.format(year, month, day, ship_num) + 'cp')
        ship_num += 1


if __name__ == '__main__':
    n_ships = 20
    convert_ais_to_sh_files(day=1, input_dir="../inputs", output_dir="../outputs")
    copy_ships(scenario_path='./MiamiPortScenario/', new_scen_path='./MiamiPortScenario_copy/', num_ships=n_ships)
