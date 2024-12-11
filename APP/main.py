from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
from close import create_plotly_figure, predict_stock, predict_stock_only
from sentiment import predict_sentiment, predict_sentiment_model
import os
import asyncio

# Membuat instance Dash
app = Dash()

# Path relatif ke file data
base_path = os.path.dirname(__file__)  # Direktori dari script yang sedang dijalankan
data_dir = os.path.join(base_path, 'data')
model_dir = os.path.join(base_path, 'Model')
method_dir = os.path.join(base_path, 'method')

# Ambil daftar file CSV yang ada di folder 'data'
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Fungsi async untuk melakukan prediksi stock
async def async_predict_stock(path):
    loop = asyncio.get_event_loop()
    # Menjalankan predict_stock di background
    train, valid, rmse, mae = await loop.run_in_executor(None, predict_stock, path)
    return train, valid, rmse, mae

# Fungsi untuk menjalankan task async secara sinkron (karena Dash tidak mendukung async callback langsung)
def get_graph_data(csv_file):
    relative_path = os.path.join(data_dir, csv_file)
    return asyncio.run(async_predict_stock(relative_path))

# Fungsi async untuk melakukan prediksi stock only
async def async_predict_stock_only(data_path, model_path):
    loop = asyncio.get_event_loop()
    # Menjalankan predict_stock_only di background
    train, valid, rmse, mae = await loop.run_in_executor(None, predict_stock_only, data_path, model_path)
    return train, valid, rmse, mae

# Fungsi untuk menjalankan task async secara sinkron (karena Dash tidak mendukung async callback langsung)
def get_graph_data_only(h5_file, csv_file):
    relative_path_model = os.path.join(model_dir, h5_file)
    relative_path = os.path.join(data_dir, csv_file)
    return asyncio.run(async_predict_stock_only(relative_path, relative_path_model))

# Fungsi async untuk melakukan prediksi sentiment
async def async_predict_sentiment(fileUrl, sentimentUrl):
    loop = asyncio.get_event_loop()
    # Menjalankan predict_sentiment di background
    train, valid, rmse, mae = await loop.run_in_executor(None, predict_sentiment, fileUrl, sentimentUrl)
    return train, valid, rmse, mae

# Fungsi untuk menjalankan task async secara sinkron (karena Dash tidak mendukung async callback langsung)
def get_graph_data_sentiment(fileUrl, sentimentUrl):
    relative_path = os.path.join(data_dir, fileUrl)
    print(f"Relative path: {relative_path}")
    print(f"Sentiment path: {sentimentUrl}")
    return asyncio.run(async_predict_sentiment(relative_path, sentimentUrl))

# Fungsi async untuk melakukan prediksi sentiment model
async def async_predict_sentiment_model(fileUrl, modelUrl, model_path):
    loop = asyncio.get_event_loop()
    # Menjalankan predict_sentiment di background
    train, valid, rmse, mae = await loop.run_in_executor(None, predict_sentiment_model, fileUrl, modelUrl, model_path)
    return train, valid, rmse, mae

# Fungsi untuk menjalankan task async secara sinkron (karena Dash tidak mendukung async callback langsung)
def get_graph_data_sentiment_model(fileUrl, modelUrl, model_path):
    relative_path = os.path.join(data_dir, fileUrl)
    relative_path_model = os.path.join(model_dir, model_path)
    print(f"Relative path: {relative_path}")
    print(f"Sentiment path: {modelUrl}")
    return asyncio.run(async_predict_sentiment_model(relative_path, modelUrl, relative_path_model))

# Fungsi callback Dash untuk memperbarui dropdown method
@app.callback(
    Output('method-dropdown', 'options'),  # Update options dari dropdown metode
    Output('method-dropdown', 'value'),   # Set default value untuk dropdown metode
    Input('csv-dropdown', 'value')        # Input pertama adalah pemilihan file CSV
)
def update_method_dropdown(selected_file):
    # Pastikan file dipilih sebelum melanjutkan
    if selected_file is None:
        return [], None  # Tidak ada pilihan file, kosongkan dropdown metode

    # Mengambil nama folder berdasarkan file yang dipilih
    folder_name = selected_file.split('_')[0].lower()  # Misalnya "bbni" dari "bbni_jk.csv"
    method_folder = os.path.join(base_path, 'method', folder_name)

    # Cek apakah folder model tersedia dan ambil daftar file di dalamnya
    if os.path.isdir(method_folder):
        method_files = [f for f in os.listdir(method_folder) if f.endswith('.csv')]  # Filter file .csv
        options = [{'label': f.split('.')[0], 'value': f} for f in method_files]
        options.insert(0, {'label': 'Close', 'value': 'close'})  # Menambahkan 'close' sebagai opsi
        return options, 'close'  # Set default ke 'close'
    else:
        # Jika folder tidak ditemukan, tetap 'close' sebagai pilihan
        return [{'label': 'Close', 'value': 'close'}], 'close'

# Fungsi callback Dash utama untuk memperbarui grafik
@app.callback(
    Output('graph-content', 'figure'),
    Output('file-name', 'children'),  # Display the name of the file selected
    Output('rmse-mae', 'children'),  # Display RMSE and MAE
    Input('csv-dropdown', 'value'),  # Dropdown input untuk memilih file CSV
    Input('method-dropdown', 'value')  # Dropdown input untuk memilih metode prediksi
)
def update_graph(selected_file, selected_method):
    # Pastikan file dan metode dipilih sebelum melanjutkan
    if selected_file is None:
        return {}, "No file or method selected"

    # Memeriksa apakah model yang sesuai ada di folder Model/
    model_file = selected_file.split('_')[0] + '_close.h5'

    # Tentukan folder model berdasarkan metode yang dipilih
    folder_name = selected_file.split('_')[0].lower()
    if(selected_method is not None):
        method_model_file = os.path.join(method_dir, folder_name, selected_method)
        print(f"Method model file: {method_model_file}")
    
    print(f"Model file: {model_file}")
    print(f"Selected method: {selected_method}")
    print(f"Selected file: {selected_file}")

    if selected_method is 'close':
        if model_file in os.listdir(model_dir):
            print(f"Model {model_file} found in Model/ folder")
            train, valid, rmse, mae = get_graph_data_only(model_file, selected_file)
            return create_plotly_figure(train, valid, rmse, mae), f"Selected File: {selected_file}", f"RMSE: {rmse:.4f}, MAE: {mae:.4f}"

        else:
            print(f"Using method {selected_method} with file {selected_file}")
            # Menunggu hasil prediksi dari fungsi async dengan menjalankan `asyncio.run()`
            train, valid, rmse, mae = get_graph_data(selected_file)
            return create_plotly_figure(train, valid, rmse, mae), f"Selected File: {selected_file}", f"RMSE: {rmse:.4f}, MAE: {mae:.4f}"


    if selected_method.split('.')[0] == 'siebert' and os.path.isfile(method_model_file):
        model_file = selected_file.split('_')[0] + '_siebert.h5'
        if model_file in os.listdir(model_dir):
            print(f"Model {model_file} found in Model/ folder")
            train, valid, rmse, mae = get_graph_data_only(model_file, selected_file, model_file)
            return create_plotly_figure(train, valid, rmse, mae), f"Selected File: {selected_file}, Method: {selected_method}", f"RMSE: {rmse:.4f}, MAE: {mae:.4f}"

        print(f"Model {model_file} found in method folder")
        train, valid, rmse, mae = get_graph_data_sentiment(selected_file, method_model_file)
        return create_plotly_figure(train, valid, rmse, mae), f"Selected File: {selected_file}, Method: {selected_method}", f"RMSE: {rmse:.4f}, MAE: {mae:.4f}"
    elif selected_method.split('.')[0] == 'manual body' and os.path.isfile(method_model_file):
        model_file = selected_file.split('_')[0] + '_manual body.h5'
        if model_file in os.listdir(model_dir):
            print(f"Model {model_file} found in Model/ folder")
            train, valid, rmse, mae = get_graph_data_sentiment_model(selected_file, method_model_file, model_file)
            return create_plotly_figure(train, valid, rmse, mae), f"Selected File: {selected_file}, Method: {selected_method}", f"RMSE: {rmse:.4f}, MAE: {mae:.4f}"

        print(f"Model {model_file} found in method folder")
        train, valid, rmse, mae = get_graph_data_sentiment(selected_file, method_model_file)
        return create_plotly_figure(train, valid, rmse, mae), f"Selected File: {selected_file}, Method: {selected_method}", f"RMSE: {rmse:.4f}, MAE: {mae:.4f}"
    
    elif model_file in os.listdir(model_dir):
        print(f"Model {model_file} found in Model/ folder")
        train, valid, rmse, mae = get_graph_data_only(model_file, selected_file)
        return create_plotly_figure(train, valid, rmse, mae), f"Selected File: {selected_file}", f"RMSE: {rmse:.4f}, MAE: {mae:.4f}"

    else:
        print(f"Using method {selected_method} with file {selected_file}")
        # Menunggu hasil prediksi dari fungsi async dengan menjalankan `asyncio.run()`
        train, valid, rmse, mae = get_graph_data(selected_file)
        return create_plotly_figure(train, valid, rmse, mae), f"Selected File: {selected_file}", f"RMSE: {rmse:.4f}, MAE: {mae:.4f}"

# Layout Dash
app.layout = html.Div([
    html.H1(children='Generative AI Stock', style={'textAlign': 'center'}),
    
    # Dropdown untuk memilih file CSV
    dcc.Dropdown(
        id='csv-dropdown',
        options=[{'label': f.split('_')[0] + '.JK', 'value': f} for f in csv_files],  # Menampilkan semua file CSV di folder 'data'
        value=csv_files[0] if csv_files else None,  # Pilih file pertama jika tersedia
        style={'width': '50%'}
    ),
    
    # Dropdown untuk memilih metode prediksi
    dcc.Dropdown(
        id='method-dropdown',
        options=[{'label': 'Close', 'value': 'close'}],  # Menambahkan 'close' sebagai opsi default
        value='close',  # Nilai default adalah 'close'
        style={'width': '50%', 'marginTop': '20px'}
    ),
    
    # Menampilkan nama file dan metode yang dipilih
    html.Div(id='file-name', style={'textAlign': 'center', 'marginTop': '20px'}),

    # Display RMSE and MAE
    html.Div(id='rmse-mae', style={'textAlign': 'center', 'marginTop': '20px'}),
    
    # Graph untuk menampilkan hasil prediksi
    dcc.Graph(id='graph-content')
])

if __name__ == '__main__':
    app.run_server(debug=False)
