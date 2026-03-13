import torch
import torch.nn as nn
import torch.nn.functional as F

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    LACF-Net: Lightweight Adaptive Channel-Fusion Network
    
    Structure:
    1. CI Branch: Shared MLP (Decomposition based)
    2. CD Branch: Top-k Mean Attention
    3. Dynamic Fusion: Gating mechanism
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # --- CI Branch (Channel Independent) ---
        # Decomposition
        self.kernel_size = 25
        self.decompsition = series_decomp(self.kernel_size)
        
        # Shared MLPs for Seasonal and Trend
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self.dropout = nn.Dropout(getattr(configs, 'dropout', 0.1))

        # --- CD Branch (Channel Dependent) ---
        # Top-k correlation
        self.top_k = getattr(configs, 'top_k', 3) 
        
        # Projection layer
        self.cd_proj = nn.Linear(self.seq_len, self.pred_len)
        
        # Channel Interaction: We use a simple linear transformation after aggregation if needed
        # or just rely on the aggregation. 
        # Adding a small MLP to refine the aggregated features
        self.cd_refine = nn.Sequential(
            nn.Linear(self.pred_len, self.pred_len),
            nn.ReLU(),
            nn.Linear(self.pred_len, self.pred_len)
        )

        
        # --- Dynamic Fusion ---
        # Gating network
        # We process the input to determine the weight [0, 1] for each channel
        # To keep it lightweight, we can use a reduced dimension for the gating input
        self.gate_dim_reduction = nn.Linear(self.seq_len, 16) 
        self.gate_fc = nn.Sequential(
            nn.Linear(16 * self.enc_in, 128),
            nn.ReLU(),
            nn.Linear(128, self.enc_in),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [Batch, Seq_Len, Channel]
        B, S, C = x.shape
        
        # ==================== CI Branch ====================
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1) # [B, C, S]
        
        seasonal_output = self.linear_seasonal(seasonal_init) # [B, C, P]
        trend_output = self.linear_trend(trend_init)       # [B, C, P]
        
        ci_output = seasonal_output + trend_output         # [B, C, P]
        ci_output = self.dropout(ci_output)

        # ==================== CD Branch ====================
        # 1. Project input (using seasonal component or raw? Raw is better for full context)
        # We use x permuted: [B, C, S]
        x_in_perm = x.permute(0, 2, 1) # [B, C, S]
        cd_proj = self.cd_proj(x_in_perm) # [B, C, P]
        
        # 2. Pearson Correlation Calculation
        # Normalize along time dimension
        mean = x_in_perm.mean(dim=2, keepdim=True)
        # Add epsilon to std to avoid division by zero
        std = x_in_perm.std(dim=2, keepdim=True) + 1e-5 
        x_norm = (x_in_perm - mean) / std # [B, C, S]
        
        # Correlation matrix: R = X_norm * X_norm^T / S
        # [B, C, S] @ [B, S, C] -> [B, C, C]
        corr_matrix = torch.matmul(x_norm, x_norm.transpose(1, 2)) / S
        
        # Handle nan/inf in correlation matrix if any (e.g. constant series)
        corr_matrix = torch.nan_to_num(corr_matrix, 0.0)

        # 3. Top-k Selection & Mean Attention
        k = min(self.top_k, C)
        # We want top-k correlated channels for each channel
        # Get indices of top-k values
        # We use absolute value of correlation as per proposal ("absolute value >= 0.3")
        # But for "Mean Attention", positive correlation helps prediction directly.
        # Strong negative correlation is also useful. We stick to correlation values for selection,
        # but maybe attend to values. Simple mean might cancel out if signs differ? 
        # Text says "Mean Attention". We will average the projected features.
        
        # Select top-k by absolute correlation
        val, indices = torch.topk(torch.abs(corr_matrix), k, dim=-1)
        
        # Create a weight matrix for gathering
        # We want to Average (Mean) the features of Top-k neighbors
        mask = torch.zeros_like(corr_matrix)
        mask.scatter_(2, indices, 1.0) # Set chosen to 1
        mask = mask / k # Normalize for Mean
        
        # Apply Aggregation
        # cd_out [B, C, P] = mask [B, C, C] @ cd_proj [B, C, P]
        cd_agg = torch.matmul(mask, cd_proj)
        
        # Refine
        cd_output = self.cd_refine(cd_agg) # [B, C, P]

        # ==================== Fusion ====================
        # Calculate Gate
        # Reduce dimensionality first to save params
        # x_in_perm: [B, C, S] -> [B, C, 16]
        x_reduced = self.gate_dim_reduction(x_in_perm) 
        # Flatten: [B, C*16]
        gate_in = x_reduced.reshape(B, -1)
        
        gate = self.gate_fc(gate_in) # [B, C]
        gate = gate.unsqueeze(-1) # [B, C, 1]
        
        # Weighted Sum
        final_output = gate * ci_output + (1 - gate) * cd_output
        
        return final_output.permute(0, 2, 1) # [B, P, C]
