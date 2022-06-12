import torch
import torchvision

# pretrained modelを読み込み
model = torchvision.models.vgg16(pretrained=True)
# modelの重みをstate_dict形式で保存
torch.save(model.state_dict(), './data/vgg16/model_weights.pth')

# modelをランダムの重みで初期化
model = torchvision.models.vgg16()
# modelの重みをload 
model.load_state_dict(torch.load('./data/vgg16/model_weights.pth'))
# dropoutやpatch normalization layerをevaluation modeに切り替える（これを忘れると正確な推論ではなくなる）
model.eval()

# modelの形ごとsave
torch.save(model, './data/vgg16/model.pth')
# modelごとloadする
model = torch.load('./data/vgg16/model.pth')