import pandas as pd
import numpy as np

x_min, x_max = -5, 5
quan_the = ['01010', '01100', '10010', '11000']
ten_ca_the = ['A', 'B', 'C', 'D']
Roulette = [0.2, 0.8, 0.5, 0.9]
lan_quay = [1, 2, 3, 4]
n = len(quan_the[0])

def bang_gia_tri_quan_the():
    df = pd.DataFrame(quan_the, columns=['Chuỗi Bit'])
    df['Thập phân'] = df['Chuỗi Bit'].apply(lambda x: int(x, 2))
    df['Giá trị thực'] = x_min + df['Thập phân'] / (2**n - 1) * (x_max - x_min)
    df['f(x)'] = np.square(df['Giá trị thực'])
    df['fitness(x)'] = 1/(1+df['f(x)'])
    df['Pk'] = df['fitness(x)']/df['fitness(x)'].sum()
    df['Qj'] = df['Pk'].cumsum()

    tong_fitness = df['fitness(x)'].sum()
    tong_Pk = df['Pk'].sum()
    row_tong = pd.DataFrame({
        'Chuỗi Bit': [''], 'Thập phân': [None], 'Giá trị thực': [None],
        'f(x)': ['Tổng'], 'fitness(x)': [tong_fitness], 'Pk': [tong_Pk], 'Qj': [None]
    })
    df_final = pd.concat([df, row_tong], ignore_index=True)
    
    print("BẢNG GIÁ TRỊ QUẦN THỂ:")
    print(df_final.round(4).fillna('').to_string(index=False))
    return df
    

df_bang_gia_tri_quan_the = bang_gia_tri_quan_the()

def bang_chon_ca_the():
    df_goc = df_bang_gia_tri_quan_the
    qj = df_goc['Qj'].tolist()
    conditions = [] 
    selected_individuals = []
    
    for i in Roulette:
        if i <= qj[0]:
            s = f"{i} < Q1"
            chon = ten_ca_the[0]
        elif qj[0] < i <= qj[1]:
            s = f"Q1 < {i} < Q2"
            chon = ten_ca_the[1]
        elif qj[1] < i <= qj[2]:
            s = f"Q2 < {i} < Q3"
            chon = ten_ca_the[2]
        elif qj[2] < i <= qj[3]:
            s = f"Q3 < {i} < Q4"
            chon = ten_ca_the[3]
        else:
            s = "Out of range"
            chon = "None"
        conditions.append(s)
        selected_individuals.append(chon)

    df = pd.DataFrame({'Lần quay Roulette': lan_quay})
    df['Xác suất Rand'] = Roulette
    df['Kiểm tra điều kiện'] = conditions 
    df['Cá thể được chọn'] = selected_individuals
    print("\nBẢNG CHỌN CÁ THỂ:")
    print(df.to_string(index=False))
    return df

df_bang_chon_ca_the = bang_chon_ca_the()

def bang_lai_ghep(diem_cat_mac_dinh=2):
    ten_sang_chuoi = dict(zip(ten_ca_the, quan_the))
    ten_duoc_chon = df_bang_chon_ca_the['Cá thể được chọn'].tolist()
    chuoi_duoc_chon = [ten_sang_chuoi.get(ten, '') for ten in ten_duoc_chon]

    ket_qua = []
    con_moi = []

    for i in range(0, len(chuoi_duoc_chon), 2):
        if i + 1 >= len(chuoi_duoc_chon):
            break

        bo_me_1 = chuoi_duoc_chon[i]
        bo_me_2 = chuoi_duoc_chon[i + 1]
        diem_cat = diem_cat_mac_dinh

        if not bo_me_1 or not bo_me_2 or len(bo_me_1) != len(bo_me_2):
            continue

        con_1 = bo_me_1[:diem_cat] + bo_me_2[diem_cat:]
        con_2 = bo_me_2[:diem_cat] + bo_me_1[diem_cat:]

        ket_qua.append({
            'Cặp': i // 2 + 1,
            'Bố/Mẹ 1': bo_me_1,
            'Bố/Mẹ 2': bo_me_2,
            'Điểm cắt': diem_cat,
            'Con 1': con_1,
            'Con 2': con_2
        })

        con_moi.extend([con_1, con_2])

    df = pd.DataFrame(ket_qua)
    print("\nBẢNG LAI GHÉP:")
    if df.empty:
        print("Không có cặp nào hợp lệ để lai ghép.")
    else:
        print(df.to_string(index=False))
        print(f"\nQuần thể con sau lai ghép: {con_moi}")
    return df, con_moi


df_bang_lai_ghep, quan_the_con = bang_lai_ghep(diem_cat_mac_dinh=2)

