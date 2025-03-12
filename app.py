import streamlit as st
import io
import base64
import ast
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem
import matplotlib.pyplot as plt
import stmol  # pip install stmol
from streamlit_ketcher import st_ketcher  # pip install streamlit-ketcher

# 문자열을 리스트로 변환하는 함수 (캐시 함수 내에서만 사용)
def convert_to_list(value):
    try:
        if value in ["None", "nan", ""]:
            return np.nan
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return np.nan

# GNPS 라이브러리 파일을 로드 및 전처리하는 함수 (캐시)
@st.cache_data
def load_gnps_library(path):
    df = pd.read_excel(path, index_col=0, engine='openpyxl')
    df["m/z_intensity"] = df["m/z_intensity"].apply(convert_to_list)
    return df

# 앱 제목 및 설명
st.title("Mass Spectrum Similarity Analysis")
st.write(
    "GNPS 라이브러리 파일은 고정 경로에서 불러오며,\n"
    "새로운 샘플 데이터를 업로드하거나 GNPS 라이브러리 내에서 검색 및 SMILES 검색 기능을 사용할 수 있습니다."
)

# 1. GNPS 라이브러리 로드 (캐시된 함수 사용)
excel_path = r"C:\TEMP\test dir\TopSpin-Jupyter\examples\GNPS_Library_pubchem_API.xlsx"
try:
    gnps_library_raw_2 = load_gnps_library(excel_path)
    st.success("GNPS 라이브러리 데이터가 성공적으로 로드되었습니다.")
except Exception as e:
    st.error("GNPS 라이브러리 파일 로드 실패: " + str(e))
    st.stop()

# 사이드바 - 입력 방식 선택
st.sidebar.header("입력 방식 선택")
input_mode = st.sidebar.radio("새로운 샘플 데이터 입력 방법", ("파일 업로드", "검색 기능", "SMILES 검색 기능"))

# (a) Top N 피크 추출 함수
def extract_top_peaks(mz_intensity_list, top_n=5):
    if not isinstance(mz_intensity_list, list) or len(mz_intensity_list) == 0:
        return None
    valid_peaks = [x for x in mz_intensity_list if isinstance(x, (tuple, list)) and len(x) >= 2]
    if not valid_peaks:
        return None
    top_peaks = sorted(valid_peaks, key=lambda x: x[1], reverse=True)[:top_n]
    return top_peaks

# GNPS 라이브러리 데이터에 대해 top_peaks 생성
gnps_library_raw_2["top_peaks"] = gnps_library_raw_2["m/z_intensity"].apply(lambda x: extract_top_peaks(x, top_n=5))

# (b) mass spectrum 유사도 계산 함수 (m/z 값은 소숫점 첫째 자리 반올림)
def mass_spectrum_similarity(peaks1, peaks2, tolerance=0.0):
    if peaks1 is None or peaks2 is None:
        return 0.0
    common_peaks = 0
    intensity_match = 0
    peaks1 = sorted([(round(x[0], 1), x[1]) for x in peaks1], key=lambda x: x[0])
    peaks2 = sorted([(round(x[0], 1), x[1]) for x in peaks2], key=lambda x: x[0])
    for mz1, intensity1 in peaks1:
        for mz2, intensity2 in peaks2:
            if abs(mz1 - mz2) <= tolerance:
                common_peaks += 1
                intensity_match += 1 - abs(intensity1 - intensity2) / max(intensity1, intensity2)
                break
    similarity_score = (common_peaks / len(peaks1)) * 0.5 + (intensity_match / len(peaks1)) * 0.5
    return similarity_score

# (c) mass spectrum 플롯을 PNG 바이트로 반환하는 함수 (dpi=600)
def plot_mass_spectrum_to_bytes(data, y_offset=10, text_offset=2, figsize=(5, 3)):
    if data is None:
        return None
    mz, intensity = zip(*data)
    intensity = np.array(intensity)
    norm_intensity = intensity / intensity.max() * 100
    fig, ax = plt.subplots(figsize=figsize)
    markerline, stemlines, baseline = ax.stem(mz, norm_intensity)
    plt.setp(markerline, marker='o', markersize=0)
    plt.setp(stemlines, linewidth=2.5, color='blue')
    plt.setp(baseline, visible=False)
    y_max = norm_intensity.max()
    ax.set_ylim(0, y_max + y_offset)
    for x, y in zip(mz, norm_intensity):
        ax.text(x, y + text_offset, f"{round(x, 1):.1f}", ha='center', va='bottom', fontsize=8, color='black')
    ax.set_xlabel('m/z', fontsize=10)
    ax.set_ylabel('Intensity', fontsize=10)
    ax.set_title('Mass Spectrum', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=600)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# 모드별 처리
if input_mode == "파일 업로드":
    st.sidebar.info("업로드 파일은 Excel 형식이며, 'm/z'와 'Intensity' 컬럼이 없으면 첫 두 컬럼을 사용합니다.")
    new_sample_file = st.sidebar.file_uploader("새로운 샘플 Excel 파일", type=["xlsx"])
    if new_sample_file is not None:
        try:
            new_sample_df = pd.read_excel(new_sample_file, engine='openpyxl')
            if 'm/z' in new_sample_df.columns and 'Intensity' in new_sample_df.columns:
                new_sample = list(new_sample_df[['m/z', 'Intensity']].itertuples(index=False, name=None))
            else:
                new_sample = list(new_sample_df.iloc[:, :2].itertuples(index=False, name=None))
            st.sidebar.success("새로운 샘플 데이터가 로드되었습니다.")
        except Exception as e:
            st.sidebar.error("새로운 샘플 파일 로드 실패: " + str(e))
            st.stop()
        new_sample_top = extract_top_peaks(new_sample, top_n=5)
        st.subheader("새로운 샘플 Mass Spectrum")
        new_sample_spectrum_bytes = plot_mass_spectrum_to_bytes(new_sample_top)
        if new_sample_spectrum_bytes is not None:
            st.image(new_sample_spectrum_bytes, width=250, caption="New Sample Mass Spectrum")
        else:
            st.write("Mass Spectrum 데이터가 없습니다.")
        
        # GNPS 라이브러리와 유사도 검색
        similarities = []
        for i, row in gnps_library_raw_2.iterrows():
            existing_peaks = row["top_peaks"]
            similarity = mass_spectrum_similarity(new_sample_top, existing_peaks, tolerance=0.0)
            similarities.append((i, similarity))
        top_10_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
        
        st.subheader("유사도 상위 10개 샘플")
        for rank, (sample_id, similarity) in enumerate(top_10_matches, start=1):
            row = gnps_library_raw_2.loc[sample_id]
            sample_name = str(row["NAME"])
            mass = row["PEPMASS"]
            smiles = row["SMILES"] if pd.notna(row["SMILES"]) else row["SMILES_re"]
            top_peaks = row["top_peaks"]
            source_instrument = row["SOURCE_INSTRUMENT"]
            if pd.isna(smiles):
                st.write(f"⚠️ {rank}위 | 샘플 ID: {sample_id} | Name: {sample_name} | Mass: {round(mass, 1)}")
                st.write(f"     Instrument: {source_instrument} | 유사도: {similarity:.3f} (SMILES 없음)")
                continue
            st.write(f"⭐ {rank}위 | 샘플 ID: {sample_id} | Name: {sample_name} | Mass: {round(mass, 1)}")
            st.write(f"     Instrument: {source_instrument} | 유사도: {similarity:.3f}")
            search_url = f"https://www.google.com/search?q={sample_name}"
            st.markdown(f"[검색 결과 보기: {sample_name}]({search_url})", unsafe_allow_html=True)
            mol = Chem.MolFromSmiles(str(smiles))
            if mol:
                Draw.MolToFile(mol, 'temp_structure.png', size=(1200, 1200))
                with open('temp_structure.png', 'rb') as f:
                    structure_bytes = f.read()
            else:
                st.write(f"⚠️ {rank}위 샘플의 SMILES 변환 실패")
                continue
            spectrum_bytes = plot_mass_spectrum_to_bytes(top_peaks)
            col1, col2 = st.columns(2)
            col1.image(structure_bytes, width=300, caption="Molecule Structure")
            col2.image(spectrum_bytes, width=400, caption="Mass Spectrum")
    else:
        st.sidebar.info("새로운 샘플 파일을 업로드해주세요.")

elif input_mode == "검색 기능":
    st.sidebar.header("샘플 검색")
    search_query = st.sidebar.text_input("검색할 샘플 이름 입력", value="")
    if search_query:
        matching_df = gnps_library_raw_2[gnps_library_raw_2["NAME"].str.contains(search_query, case=False, na=False)]
        if matching_df.empty:
            st.sidebar.info("검색 결과가 없습니다.")
        else:
            st.sidebar.success(f"{len(matching_df)}개의 결과가 검색되었습니다.")
            selected_sample_id = st.sidebar.selectbox("검색 결과에서 샘플 선택", matching_df.index)
            selected_row = gnps_library_raw_2.loc[selected_sample_id]
            st.subheader(f"검색 결과: {str(selected_row['NAME'])}")
            st.write(f"Mass: {selected_row['PEPMASS']}")
            st.write(f"Source Instrument: {selected_row['SOURCE_INSTRUMENT']}")
            selected_top_peaks = selected_row["top_peaks"]
            spectrum_bytes = plot_mass_spectrum_to_bytes(selected_top_peaks)
            st.image(spectrum_bytes, width=400, caption="Mass Spectrum")
            smiles = selected_row["SMILES"] if pd.notna(selected_row["SMILES"]) else selected_row["SMILES_re"]
            if smiles:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol:
                    Draw.MolToFile(mol, "temp_structure.png", size=(1200, 1200))
                    with open("temp_structure.png", "rb") as f:
                        structure_bytes = f.read()
                    st.image(structure_bytes, width=300, caption="Molecule Structure")
                else:
                    st.write("유효하지 않은 SMILES")
            else:
                st.write("SMILES 정보가 없습니다.")
    else:
        st.sidebar.info("검색어를 입력해주세요.")

elif input_mode == "SMILES 검색 기능":
    st.sidebar.header("SMILES 검색 기능")
    st.sidebar.info("직접 그린 분자 구조의 SMILES 코드를 이용하여 정확히 일치하는 샘플을 검색합니다.\n아래 Ketcher를 사용하여 분자 구조를 편집하세요.")
    DEFAULT_MOL = "OC1=C(O)C=C(/C=C/C(O)=O)C=C1"
    # 분자 구조 편집기: st_ketcher를 사용하여 SMILES 코드를 편집합니다.
    molecule = st.text_input("Molecule", DEFAULT_MOL)
    try:
        smile_code = st_ketcher(molecule)
    except Exception as e:
        st.warning("st_ketcher 기능 사용 불가. 기본 입력값을 사용합니다.")
        smile_code = molecule
    st.markdown(f"편집된 SMILES: ``{smile_code}``")
    
    if smile_code:
        try:
            query_mol = Chem.MolFromSmiles(smile_code)
            if query_mol is None:
                st.error("유효하지 않은 SMILES 코드입니다.")
            else:
                query_canonical = Chem.MolToSmiles(query_mol, canonical=True)
                found = False
                for i, row in gnps_library_raw_2.iterrows():
                    sample_smiles = row["SMILES"] if pd.notna(row["SMILES"]) else row["SMILES_re"]
                    if sample_smiles:
                        sample_mol = Chem.MolFromSmiles(str(sample_smiles))
                        if sample_mol:
                            sample_canonical = Chem.MolToSmiles(sample_mol, canonical=True)
                            if sample_canonical == query_canonical:
                                found = True
                                st.subheader("SMILES 검색 결과")
                                st.write(f"일치하는 샘플: {str(row['NAME'])}")
                                st.write(f"Mass: {row['PEPMASS']}")
                                st.write(f"Source Instrument: {row['SOURCE_INSTRUMENT']}")
                                top_peaks = row["top_peaks"]
                                spectrum_bytes = plot_mass_spectrum_to_bytes(top_peaks)
                                if spectrum_bytes is not None:
                                    st.image(spectrum_bytes, width=400, caption="Mass Spectrum")
                                else:
                                    st.write("Mass Spectrum 데이터가 없습니다.")
                                sample_smiles_str = str(sample_smiles)
                                sample_mol = Chem.MolFromSmiles(sample_smiles_str)
                                if sample_mol:
                                    Draw.MolToFile(sample_mol, "temp_structure.png", size=(1200, 1200))
                                    with open("temp_structure.png", "rb") as f:
                                        structure_bytes = f.read()
                                    st.image(structure_bytes, width=300, caption="Molecule Structure")
                                search_url = f"https://www.google.com/search?q={str(row['NAME'])}"
                                st.markdown(f"[검색 결과 보기: {str(row['NAME'])}]({search_url})", unsafe_allow_html=True)
                                break
                if not found:
                    st.error("입력한 SMILES와 정확히 일치하는 샘플이 없습니다.")
        except Exception as e:
            st.error(str(e))
    else:
        st.sidebar.info("SMILES를 입력해주세요.")
