import torch  
import torch.nn as nn  
  
def linf_bim_attack(  
    model: nn.Module,  
    x: torch.Tensor,           # 入力画像（モデルの境界内）  
    labels: torch.Tensor,      # 真のラベル  
    epsilon: float,            # 摂動の最大値（モデルの境界空間で）  
    rel_stepsize: float = 0.2, # epsilonに対する相対ステップサイズ  
    steps: int = 10,           # 反復回数  
    random_start: bool = False,  
    bounds: tuple = (0, 1),    # モデルの境界  
) -> torch.Tensor:  
    """  
    L∞ Basic Iterative Method  
      
    Args:  
        model: PyTorchモデル  
        x: 入力画像（既にモデルの境界空間に正規化済み）  
        labels: 真のラベル  
        epsilon: 摂動の最大値（モデルの境界空間で指定）  
        rel_stepsize: epsilonに対する相対ステップサイズ  
        steps: 反復回数  
        random_start: ランダム初期化するか  
        bounds: モデルの入力境界（例: (0, 1)または(0, 255)）  
    """  
    min_bound, max_bound = bounds  
      
    # ステップサイズの計算  
    stepsize = rel_stepsize * epsilon  
      
    # 初期化  
    x_adv = x.clone().detach()  
      
    if random_start:  
        # epsilon球内でランダムに初期化  
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)  
        x_adv = torch.clamp(x_adv, min_bound, max_bound)  
      
    # 反復的な攻撃  
    for step in range(steps):  
        x_adv.requires_grad = True  
          
        # 順伝播  
        logits = model(x_adv)  
          
        # 損失計算（クロスエントロピー）  
        loss = nn.CrossEntropyLoss()(logits, labels)  
          
        # 勾配計算  
        model.zero_grad()  
        loss.backward()  
        grad = x_adv.grad.detach()  
          
        with torch.no_grad():  
            # 勾配の正規化：L∞の場合はsign()を使用  
            normalized_grad = grad.sign()  
              
            # 勾配方向に更新  
            x_adv = x_adv + stepsize * normalized_grad  
              
            # epsilon球への射影  
            perturbation = torch.clamp(x_adv - x, -epsilon, epsilon)  
            x_adv = x + perturbation  
              
            # モデルの境界への射影  
            x_adv = torch.clamp(x_adv, min_bound, max_bound)  
      
    return x_adv


import torch  
import torch.nn as nn  
  
def linf_bim_attack_with_normalization(  
    model: nn.Module,  
    x: torch.Tensor,              # 入力画像（任意の範囲、例: [0, 255]）  
    labels: torch.Tensor,         # 真のラベル  
    epsilon: float,               # 摂動の最大値（入力画像と同じ空間で指定）  
    input_bounds: tuple = (0, 255),   # 入力画像の境界  
    model_bounds: tuple = (0, 1),     # モデルが期待する境界  
    rel_stepsize: float = 0.2,  
    steps: int = 10,  
    random_start: bool = False,  
) -> torch.Tensor:  
    """  
    境界変換を含むL∞ BIM実装  
      
    Args:  
        model: PyTorchモデル（model_bounds空間の入力を期待）  
        x: 入力画像（input_bounds空間）  
        labels: 真のラベル  
        epsilon: 摂動の最大値（input_bounds空間で指定）  
        input_bounds: 入力画像の境界（例: (0, 255)）  
        model_bounds: モデルが期待する境界（例: (0, 1)）  
        rel_stepsize: epsilonに対する相対ステップサイズ  
        steps: 反復回数  
        random_start: ランダム初期化するか  
    """  
    input_min, input_max = input_bounds  
    model_min, model_max = model_bounds  
      
    # 入力をモデルの境界空間に変換  
    def to_model_space(x_input):  
        # [input_min, input_max] -> [model_min, model_max]  
        x_normalized = (x_input - input_min) / (input_max - input_min)  
        return x_normalized * (model_max - model_min) + model_min  
      
    # モデルの境界空間から入力空間に変換  
    def to_input_space(x_model):  
        # [model_min, model_max] -> [input_min, input_max]  
        x_normalized = (x_model - model_min) / (model_max - model_min)  
        return x_normalized * (input_max - input_min) + input_min  
      
    # epsilonをモデル空間に変換  
    epsilon_model = epsilon * (model_max - model_min) / (input_max - input_min)  
    stepsize_model = rel_stepsize * epsilon_model  
      
    # 入力をモデル空間に変換  
    x_model = to_model_space(x)  
    x_adv_model = x_model.clone().detach()  
      
    if random_start:  
        # epsilon球内でランダムに初期化（モデル空間で）  
        x_adv_model = x_adv_model + torch.empty_like(x_adv_model).uniform_(  
            -epsilon_model, epsilon_model  
        )  
        x_adv_model = torch.clamp(x_adv_model, model_min, model_max)  
      
    # 反復的な攻撃（モデル空間で実行）  
    for step in range(steps):  
        x_adv_model.requires_grad = True  
          
        # 順伝播  
        logits = model(x_adv_model)  
          
        # 損失計算  
        loss = nn.CrossEntropyLoss()(logits, labels)  
          
        # 勾配計算  
        model.zero_grad()  
        loss.backward()  
        grad = x_adv_model.grad.detach()  
          
        with torch.no_grad():  
            # 勾配の正規化：L∞の場合はsign()  
            normalized_grad = grad.sign()  
              
            # 更新  
            x_adv_model = x_adv_model + stepsize_model * normalized_grad  
              
            # epsilon球への射影（モデル空間で）  
            perturbation = torch.clamp(  
                x_adv_model - x_model,   
                -epsilon_model,   
                epsilon_model  
            )  
            x_adv_model = x_model + perturbation  
              
            # モデル境界への射影  
            x_adv_model = torch.clamp(x_adv_model, model_min, model_max)  
      
    # 結果を入力空間に戻す  
    x_adv_input = to_input_space(x_adv_model)  
    return x_adv_input