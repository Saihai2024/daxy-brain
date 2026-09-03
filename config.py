"""
项目配置
"""
import pickle
import torch
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

################# classify 相关的配置 ###############
predict_ratio = 0.98  #预测可能性的阈值


################# chatbot相关的配置 #################
chatbot_train_batch_size = 800
chatbot_test_batch_size = 500

if os.path.exists("brain/models/ws_input.pkl"):
    input_ws = pickle.load(open("brain/models/ws_input.pkl","rb"))
    target_ws = pickle.load(open("brain/models/ws_target.pkl","rb"))
else:
    input_ws = None
    target_ws = None
chatbot_input_max_len = 50
chatbot_target_max_len = 50

chatbot_encoder_embedding_dim = 300
chatbot_encoder_hidden_size = 128
chatbot_encoder_number_layer = 2
chatbot_encoder_bidirectional = True
chatbot_encoder_dropout = 0.3

chatbot_decoder_embedding_dim = 300
chatbot_decoder_hidden_size = 128*2
chatbot_decoder_number_layer = 1
chatbot_decoder_dropout = 0

beam_width = 50

# Transformer Hyperparameters (matches saved checkpoint)
transformer_d_model = 256
transformer_nhead = 8
transformer_num_layers = 2
transformer_dim_feedforward = 512
transformer_dropout = 0.1


############### 排序模型的参数############
dnnsort_max_len = 30
dnn_train_batch_size = 256
dnn_test_batch_size = 500

if os.path.exists("dnnsort/models/ws.pkl"):
    dnn_ws = pickle.load(open("dnnsort/models/ws.pkl","rb"))
else:
    dnn_ws = None
dnnsort_hidden_size = 128
dnnsort_number_layers = 2
dnnsort_bidriectional = True
dnnsort_lstm_dropout = 0.5





