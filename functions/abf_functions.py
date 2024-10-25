import os
import pandas as pd
import pyabf
import struct
import re
import dask.dataframe as dd
import matplotlib.pyplot as plt

#----READ FILES-------------------------------------------------------------------------------------------------
def read_protocol_comments(file_path):
    # Читання протоколу з файлу
    with open(file_path, 'rb') as f:
        # Знаходимо початок протоколу
        f.seek(76)
        protocol_start = struct.unpack('I', f.read(4))[0]
        f.seek(protocol_start)
        
        # Читаємо 2048 байтів даних протоколу
        protocol_data = f.read(2048)
        protocol_str = protocol_data.decode('utf-8', errors='ignore')

        # Пошук коментарів, що йдуть після "AXENGN 2.0.1.69"
        comments = []
        search_marker = "AXENGN 2.0.1.69"
        parts = protocol_str.split(search_marker)

        for part in parts[1:]:
            # Шукаємо коментарі у вигляді #{число} CM
            match = re.search(r'#\d+ CM[^\n]*', part)
            if match:
                comment = match.group().strip()
                
                # Видалення квадратних дужок
                comment = re.sub(r'[\[\]]', '', comment)
                
                # Обрізаємо коментар після появи неприпустимих символів
                comment = re.split(r'[^\x20-\x7E]|[ ]{3,}', comment)[0]
                
                # Обрізаємо зайві пробіли і неприпустимі символи
                comment = re.sub(r'\s*([^\x20-\x7E]|[ ]{3,})$', '', comment)
                
                # Додаємо коментар тільки якщо він не порожній
                if comment:
                    comments.append(comment)
    
    # Повертаємо коментарі як один рядок, об'єднаний через кому
    return ', '.join(comments)
        
        
def read_abf_files(main_directory):
    all_data = []
    for folder_name in os.listdir(main_directory):
        folder_path = os.path.join(main_directory, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.abf'):
                    abf_file_path = os.path.join(folder_path, file_name)
                    abf = pyabf.ABF(abf_file_path)
                    comments = abf.tagComments if abf.tagComments else "No comments"
                    
                    # Викликаємо функцію для зчитування коментарів з протоколу
                    protocol_comments = read_protocol_comments(abf_file_path)
                    
                    # Створюємо список для зберігання епох для кожного моменту часу
                    epoch_list = [None] * len(abf.sweepX)

                    # Витягуємо інформацію про епохи для кожного інтервалу часу
                    for i, (p1, p2) in enumerate(zip(abf.sweepEpochs.p1s, abf.sweepEpochs.p2s)):
                        epochType = abf.sweepEpochs.types[i]
                        # Призначаємо тип епохи кожній точці часу в інтервалі p1 - p2
                        for time_idx in range(p1, p2):
                            if time_idx < len(epoch_list):
                                epoch_list[time_idx] = epochType
                           
                    for sweep in range(abf.sweepCount):
                        abf.setSweep(sweep)
                        sweep_data = abf.sweepY
                        sweep_time = abf.sweepX
                        sweep_command = abf.sweepC
                        
                        # Зберігаємо дані для кожного значення в sweep'і
                        for idx, (t, data_point, command_point) in enumerate(zip(sweep_time, sweep_data, sweep_command)):
                            epoch = epoch_list[idx] if idx < len(epoch_list) else None
                            all_data.append([
                                folder_name, 
                                file_name, 
                                sweep + 1,  # To make sweeps 1-based
                                t, 
                                data_point, 
                                command_point,
                                epoch, 
                                comments,
                                protocol_comments
                            ])

    
    # Створюємо DataFrame з отриманих даних
    data_df = pd.DataFrame(all_data, columns=[
        "Folder", 
        "File", 
        "Sweep", 
        "Time (s)", 
        "Data (ADC)", 
        "Command (DAC)",
        "Epoch", 
        "Comments", 
        "Protocol Comments"
    ])
    
    return data_df
    
#----EXTRACT INFO FROM COMMENTS-------------------------------------------------------------------------------------------------

# Функція для витягу '№ Cell' та 'Cell type'
def extract_cell_info(protocol_comments):
    match = re.search(r'#(\d+)\s(\w+)', protocol_comments)
    if match:
        return match.group(1), match.group(2)
    return None, None

# Функція для витягу 'Stage' з інтервалами
def extract_stages(protocol_comments, sweep):
    stages = []
    matches = re.findall(r'(\w+\s*\w*)\s*=\s*\[(\d+):(\d*)\]', protocol_comments)
    
    for stage, start, end in matches:
        start, end = int(start), int(end) if end else None
        if end is None:  # Якщо кінець не визначено, заповнювати до кінця
            end = float('inf')  # Використати нескінченність для позначення необмеженого кінця
        if start <= sweep <= end:
            stages.append(stage.strip())
    
    return ', '.join(stages) if stages else None

#----DELETE BAD SWEEPS-------------------------------------------------------------------------------------------

def delete_sweeps(data, input_fields):
    delete_ranges = {}
    errors = []
    for file, input_field in input_fields.items():
        input_value = input_field.value.strip()
        if input_value:
            try:
                # Перевіряємо правильність введеного формату
                for part in input_value.split(','):
                    part = part.strip()
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        if start > end:
                            raise ValueError
                    else:
                        int(part)
                delete_ranges[file] = input_value
            except ValueError:
                errors.append(f"Неправильний формат для файлу: {file}. Уведіть у формі 1-3, 5.")
    
    # Якщо є помилки, виводимо повідомлення і не виконуємо видалення
    if errors:
        for error in errors:
            print(error)
        return

    # Процес видалення свіпів
    for file, ranges in delete_ranges.items():
        sweeps_to_delete = set()
        for part in ranges.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                sweeps_to_delete.update(range(start, end + 1))
            else:
                sweeps_to_delete.add(int(part))

        # Перевіряємо, чи існують введені свіпи в даних
        available_sweeps = set(data[data['File'] == file]['Sweep'])
        non_existing_sweeps = sweeps_to_delete - available_sweeps
        if non_existing_sweeps:
            print(f"Для файлу {file} таких свіпів не існує: {non_existing_sweeps}")
            continue
        
        # Видаляємо рядки
        data.drop(data[(data['File'] == file) & (data['Sweep'].isin(sweeps_to_delete))].index, inplace=True)
    
    print("Оновлені дані з видаленими свіпами:")
    print(data)

#----SAVE TO FILE-------------------------------------------------------------------------------------------------

def save_data_to_csv(data, file_name):
    # Перетворюємо DataFrame у Dask DataFrame
    ddf = dd.from_pandas(data, npartitions=1)  # Вказуємо npartitions=1 для збереження в одному CSV

    # Шлях до виходу
    csv_output_path = os.path.join(os.getcwd(), '..', 'outputs', f'{file_name}.csv')

    # Збереження даних у CSV
    ddf.to_csv(csv_output_path, index=False, single_file=True)  # Використовуємо single_file для збереження в один файл

    print(f'Data saved to {csv_output_path}')
    
    
def save_data_with_dask(data, file_name, max_rows_per_sheet=500000):
    # Перетворюємо DataFrame у Dask DataFrame
    ddf = dd.from_pandas(data, npartitions=len(data) // max_rows_per_sheet + 1)

    # Шлях до виходу
    excel_output_path = os.path.join(os.getcwd(), '..', 'outputs', f'{file_name}.xlsx')

    # Відкриваємо Excel Writer
    with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
        for i in range(ddf.npartitions):
            # Витягуємо підмножину даних
            subset_data = ddf.get_partition(i).compute()  # Повертаємо в pandas DataFrame
            sheet_name = f'Sheet{i + 1}'
            subset_data.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f'Saved: {sheet_name}')

    print(f'Data saved to {excel_output_path}')
    
#----PLOTTING-------------------------------------------------------------------------------------------------
        
def plot_grouped_by_file(data):
    # Створюємо словник для групування даних по іменам файлів
    file_groups = {}
    
    # Проходимо через всі рядки даних і групуємо по імені файлу
    for row in data:
        folder_name, file_name, sweep, sweep_time, sweep_data, sweep_command, comments, protocol_comments = row
        
        if file_name not in file_groups:
            file_groups[file_name] = []
        file_groups[file_name].append((sweep, sweep_time, sweep_data, comments, protocol_comments))
    
    # Побудова графіків по кожній групі
    for file_name, sweeps in file_groups.items():
        plt.figure(figsize=(10, 6))
        
        for sweep, sweep_time, sweep_data, comments, protocol_comments in sweeps:
            plt.plot(sweep_time, sweep_data, label=f"Sweep {sweep}")
        
        # Додаємо назву файлу та коментар протоколу як заголовок графіку
        plt.title(f"File: {file_name}\nProtocol Comments: {protocol_comments}")
        plt.xlabel("Time (s)")
        plt.ylabel("Data (ADC)")
        plt.legend()
        
        # Виведення графіку
        plt.show()
