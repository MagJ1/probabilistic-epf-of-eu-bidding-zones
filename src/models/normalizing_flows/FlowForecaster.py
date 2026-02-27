from typing import Union
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import normflows as nf
from models.normalizing_flows.normflow_wrappers import ActNormNoCtx, GaussianMixtureNoCtx

class FlowForecaster(pl.LightningModule):
    """
    Transformer encoder–decoder that outputs context vectors which condition
    a Masked Autoregressive Flow (MAF) from `normflows` to model the joint
    distribution of the next forecast_horizon electricity prices.

    Two types of forecasts are possible: 
        realistic_mode = False, where it is assumed, that at step t_0 prices and some covariates (c_future_unknown) are only known in the past. Some other covariates (c_future_known) are known in the future, i.e. calendar features. This type exists to enable better comparison with foundational models, where the requirements of the realistic_mode=True are more difficult to implement. 

        realistic_mode = True, where the model forecasts under realistic conditions. That is, that c_future_unknown covariates are only known up to t_-3, because they have to be aggregated, which takes roughly two hours. Meanwhile, prices of the steps t_0:t_11 are already known and can be used for prediction. These information, correctly masked, together with the c_future_known covariates are fed into the encoder. The masking of the c_future_unknown covariates is determined by the variable enc_unknown_cutoff. To improve the ability of the decoder, some of the last prices and covariates are also fed into it. How many steps of those the decoder can access is determined by the variable "dec_known_past_injection_horizon". 

        Example of realistic_mode = True: The model has to predict at day d_0 the 24 prices of day d_1. This is usually done close at 12:00 at noon. c_future_unknown covariates are only known until 10:00. Meanwhile, electricity prices are known in the future of d_0, since they were build at the last day d_-1. Hence, prices of steps t_0:t_11 are known. c_future_known covariates are known anyways. For the model, it looks like the forecast is done at d_1 at step t_12. This should be considered when determining the context_length. 

    """
    
    def __init__(self,
                tf_in_size: int,
                nf_hidden_dim: int,
                n_layers: int,
                n_heads: int,
                n_flow_layers: int,
                n_made_blocks: int,
                tf_dropout: float,
                c_future_unknown: int,        # past-only covariates 
                c_future_known: int,          # known-ahead covariates
                context_length: int,          
                forecast_horizon: int,        
                enc_unknown_cutoff: int,
                dec_known_past_injection_horizon: int,
                realistic_mode: bool,
                ):
        super().__init__()
        """Constructor

        Args:
            tf_in_size: Input dimension of transformer
            nf_hidden_dim: Number of hidden units in the MADE network, used by the flow
            n_layers: Number of layers of the encoder and decoder
            n_heads: Number of attention heads used by each encoder and decoder layer
            n_flow_layers: Number of flow layers
            c_future_unknown: Number of covariates, which are only known in the past (e.g. load)
            c_future_known: Number of covariates, which are also known in the future (e.g. weekday)
            context_length: Steps that the model can see in the past
            forecast_horizon: Steps the model has to predict into the future
            enc_unknown_cutoff: Steps that are masked to the encoder of those variables known only in the past
            dec_known_past_injection_horizon: Steps of the price and future known variables, that are also fed into the decoder
            realistic_mode: Toggle realistic_mode, see description above.  
        """

        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.c_future_unknown = c_future_unknown
        self.c_future_known = c_future_known
        self.enc_unknown_cutoff = enc_unknown_cutoff
        self.dec_known_past_injection_horizon = dec_known_past_injection_horizon
        self.realistic_mode = realistic_mode

        if self.realistic_mode and self.dec_known_past_injection_horizon > self.context_length:
            raise ValueError(
                f"dec_known_past_injection_horizon={self.dec_known_past_injection_horizon} "
                f"exceeds context_length={self.context_length}"
            )

        # ---------- 1. Input projections -----------------------------------
        dim_encoder = 1 + c_future_unknown + c_future_known
        dim_decoder = 1 + c_future_unknown + c_future_known   # unknown future=0
        self.enc_proj = nn.Linear(dim_encoder, tf_in_size)
        self.dec_proj = nn.Linear(dim_decoder, tf_in_size)

        # ---------- 2. Positional encoding ---------------------------------
        K = self.dec_known_past_injection_horizon if self.realistic_mode else 0
        max_total_position_len = self.context_length + self.forecast_horizon + K

        # self.pos_enc = nn.Parameter(
        #     torch.randn(1, max_total_position_len, tf_in_size) * 0.01)

        # to improve robustness of training
        self.pos_enc = nn.Parameter(torch.zeros(1, max_total_position_len, tf_in_size))

        # ---------- 3. Transformer encoder & decoder -----------------------
        enc_layer = nn.TransformerEncoderLayer(
            tf_in_size, n_heads, 4 * tf_in_size, batch_first=True, dropout=tf_dropout, norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(
            tf_in_size, n_heads, 4 * tf_in_size, batch_first=True, dropout=tf_dropout, norm_first=True)

        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, n_layers)

        self.readout = nn.Sequential(
            nn.LayerNorm(tf_in_size),
            nn.Linear(tf_in_size, tf_in_size),
            nn.Tanh(),
            nn.Linear(tf_in_size, 1)   # score per step
        )

        # ---------- 4. Normalizing flow head (MAF) -------------------------
        transforms = []
        for _ in range(n_flow_layers):
            transforms.append(
                nf.flows.MaskedAffineAutoregressive(
                    features=self.forecast_horizon,
                    hidden_features=nf_hidden_dim,
                    context_features=tf_in_size,
                    activation=torch.tanh,
                    num_blocks=n_made_blocks,
                    # has to be true, otherwise throughs NotImplementedError
                    use_residual_blocks=True,
                    use_batch_norm=False, 
                    random_mask=False
                    ))
            transforms.append(ActNormNoCtx(shape=(self.forecast_horizon,)))
            transforms.append(nf.flows.Permute(self.forecast_horizon))


        # transforms = []
        # for i in range(n_flow_layers):
        #     # Alternate which half is transformed each layer
        #     layer = nf.flows.CoupledRationalQuadraticSpline(
        #         num_input_channels=self.forecast_horizon,
        #         num_blocks=n_made_blocks,          # depth of the conditioner MLP (ResidualNet)
        #         num_hidden_channels=nf_hidden_dim, # width of conditioner
        #         num_context_channels=tf_in_size,   # <-- your T5 context size
        #         num_bins=4,                        # try 8–16
        #         tails="linear",                    # linear tails for prices
        #         tail_bound=3.0,                    # typical; 2–5 works
        #         activation=nn.ReLU,
        #         dropout_probability=0.0,
        #         reverse_mask=bool(i % 2)           # flip mask every layer
        #     )
        #     transforms.append(layer)
        #     transforms.append(ActNormNoCtx(shape=(self.forecast_horizon,)))       # keeps scales sane
        #     transforms.append(nf.flows.Permute(self.forecast_horizon))            # or learned 1x1 conv if available
        

        n_modes = 5
        dim = self.forecast_horizon

        loc     = np.random.randn(n_modes, dim).astype(np.float32)
        scale   = np.ones((n_modes, dim), dtype=np.float32)
        weights = np.ones(n_modes, dtype=np.float32)

        prior = GaussianMixtureNoCtx(
            n_modes=n_modes,
            dim=dim,
            loc=loc,                # float32
            scale=scale,            # float32
            weights=weights,        # float32
            trainable=True,
        )
        # nf.distributions.GaussianMixture() to try or standard DiagGaussian
        # prior = nf.distributions.DiagGaussian(self.forecast_horizon)
        self.flow = nf.ConditionalNormalizingFlow(prior, transforms)

        def _tiny_init(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=1e-5)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        self.flow.apply(_tiny_init)

    def _build_encoder_input(self, batch: dict) -> torch.Tensor:
        """Builds input for the encoder

        Args:
            batch:
                "ds": (B,)
                "unique_id": (B,)
                "y_past": (B,context_length)
                "c_ctx_future_unknown": (B,context_length,c_future_unknown)
                "ctx_future_known": (B,context_length,c_future_known)
                "c_fct_future_known": (B,forecast_horizon,c_future_known)
                "y_future": (B,forecast_horizon)
                
        Returns:
            enc_in: shape (B,context_length,1 + c_future_unknowns + c_future_known)
        """

        c_ctx_future_unknown = batch["c_ctx_future_unknown"]

        if self.realistic_mode:
            c_ctx_future_unknown = c_ctx_future_unknown.clone()
            c_ctx_future_unknown[:,-self.enc_unknown_cutoff:] = 0

        enc_in = torch.cat([
            batch["y_past"].unsqueeze(-1),            # (B,context_length,1)
            c_ctx_future_unknown,                     # (B,context_length,c_future_unknown)
            batch["c_ctx_future_known"]               # (B,context_length,c_future_known)
        ], dim=-1)                                    # (B,context_length,Din_enc)

        return enc_in

    def _build_decoder_input(self, batch: dict, teacher_force:bool) -> torch.Tensor:
        """Builds input for the decoder

        Args:
            batch:
                "ds": (B,)
                "unique_id": (B,)
                "y_past": (B,context_length)
                "c_ctx_future_unknown": (B,context_length,c_future_unknown)
                "ctx_future_known": (B,context_length,c_future_known)
                "c_fct_future_known": (B,forecast_horizon,c_future_known)
                "y_future": (B,forecast_horizon)
            teacher_force: Whether decoder can see future prices as well during training
                
        Returns:
            dec_in: shape (B,
                            forecast_horizon [+len(dec_known_past_injection_horizon), if realistic_mode=True],
                            1 + c_future_known + c_future_unknown)
        """
        K = self.dec_known_past_injection_horizon
        c_fct_future_known = batch["c_fct_future_known"]

        if teacher_force:                       # training mode
            y_seed = batch["y_future"]          # ground-truth future targets
        else:                                   # inference mode
            y_seed = torch.zeros_like(batch["y_future"])  # zeros as BOS

        if self.realistic_mode and K > 0:
            known_past_prices = batch["y_past"][:, -K:]     # (B,K)
            y_seed = torch.cat([known_past_prices, y_seed], dim=1)
            
            known_past_covariates = batch["c_ctx_future_known"][:,self.context_length-K:]   # (B,K,F_ck)
            c_fct_future_known = torch.cat([c_fct_future_known, known_past_covariates], dim=1)
            
        decoder_length = y_seed.size(1)
        masked_c_future_unknown = torch.zeros(
                        batch["y_past"].size(0), 
                        decoder_length, 
                        self.c_future_unknown, 
                        dtype=batch["y_past"].dtype, 
                        device=batch["y_past"].device)
        
        dec_in = torch.cat([
            # (B,[if realistic_mode: + K] forecast_horizon,1)
            y_seed.unsqueeze(-1),      
            # (B,[if realistic_mode: + K] forecast_horizon,c_future_known)
            c_fct_future_known,
            # (B,[if realistic_mode: + K] forecast_horizon,c_future_unknown)        
            masked_c_future_unknown
        ], dim=-1)

        # (B,forecast_horizon,Din_dec)
        return dec_in
    
    def make_ctx(self, batch: dict, teacher_force: bool) -> torch.Tensor:
        """ Make context for the normalizing flow model.

        Args:
            batch:
                "ds": (B,)
                "unique_id": (B,)
                "y_past": (B,context_length)
                "c_ctx_future_unknown": (B,context_length,c_future_unknown)
                "ctx_future_known": (B,context_length,c_future_known)
                "c_fct_future_known": (B,forecast_horizon,c_future_known)
                "y_future": (B,forecast_horizon)
            teacher_force: Whether decoder can see future prices as well during training

        Returns:
            dec_out: (B,forecast_horizon,tf_in_size)
        """
        if self.realistic_mode:
            K = self.dec_known_past_injection_horizon
            if not (0 <= K <= self.context_length):
                raise ValueError(
                    f"dec_known_past_injection_horizon={K} out of range [0, {self.context_length}]"
                )
        else:
            K = 0  # no injection in non-realistic mode

        # encoder
        enc_in  = self._build_encoder_input(batch)
        enc_emb = self.enc_proj(enc_in) + self.pos_enc[:, :self.context_length]
        enc_out = self.encoder(enc_emb)

        # decoder inputs
        dec_in = self._build_decoder_input(batch, teacher_force)
        dec_len = dec_in.size(1)

        # positional enc for decoder: shift start only if we actually injected
        dec_start_idx = (self.context_length - K) if (self.realistic_mode and K > 0) \
                        else self.context_length
        dec_emb = self.dec_proj(dec_in) + self.pos_enc[:, dec_start_idx:dec_start_idx + dec_len]

        # causal/self-attention mask
        if self.realistic_mode and K > 0:
            tgt_mask = torch.full((dec_len, dec_len), float("-inf"), device=self.device)
            # known-past block fully visible
            tgt_mask[:K, :K] = 0.0
            # future can attend to known past
            tgt_mask[K:, :K] = 0.0
            # autoregressive on the H future positions
            tgt_mask[K:, K:] = nn.Transformer.generate_square_subsequent_mask(
                self.forecast_horizon, device=self.device
            )
        else:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                self.forecast_horizon, device=self.device
            )

        # quick sanity: no row fully -inf
        if torch.isinf(tgt_mask).all(dim=1).any():
            bad_rows = torch.isinf(tgt_mask).all(dim=1).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(f"tgt_mask has rows with all -inf at positions: {bad_rows}")

        # run decoder
        dec_out = self.decoder(tgt=dec_emb, memory=enc_out, tgt_mask=tgt_mask)      # (B,H,tf_in_size)

        # attention pooling over horizon -> context vector for the flow
        scores = self.readout(dec_out).squeeze(-1)   # (B, H or K+H)
        attn   = scores.softmax(dim=-1).unsqueeze(-1)
        ctx_series = (dec_out * attn).sum(dim=1)     # (B, tf_in_size)

        if torch.isnan(ctx_series).any():
            bad = torch.isnan(ctx_series).any(dim=-1).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(f"ctx_series contains NaNs at batch positions: {bad}")

        return ctx_series                      

    # ---------- sampling helper (outside Lightning loop) ------------------
    # @torch.no_grad()
    def sample(self, batch: dict, n_per_series: int = 10, track_grad: bool = False)->Union[torch.Tensor, float]:
        """ Draw "n_per_series" scenarios for every series in batch.
        Args:
            batch:
                "ds": (B,)
                "unique_id": (B,)
                "y_past": (B,context_length)
                "c_ctx_future_unknown": (B,context_length,c_future_unknown)
                "ctx_future_known": (B,context_length,c_future_known)
                "c_fct_future_known": (B,forecast_horizon,c_future_known)
                "y_future": (B,forecast_horizon)
            n_per_series: Number of scenarios drawn per time series in the batch
        Returns:
            samples :  (n_per_series, B, H)
            log_q   :  (n_per_series, B)
        """
        with torch.set_grad_enabled(track_grad):
            B= batch["y_past"].size(0)

            # holds per TS of batch one context vector of length tf_in_size
            ctx_series = self.make_ctx(batch, teacher_force=False)   # (B, tf_in_size)

            S = n_per_series
            # repeat each context vector S times, s.t. the model can generate S different scenarios
            ctx_flat = ctx_series.repeat_interleave(S, dim=0)                  # (S·B,tf_in_size)

            # ---- 3. draw S·B latent vectors in a single call ------------
            x_flat, log_q_flat = self.flow.sample(S * B, context=ctx_flat)     # (S·B,forecast_horizon)

            # ---- 4. reshape back to (S, B, H) ---------------------------
            samples: torch.Tensor = x_flat.view(S, B, self.forecast_horizon)
            log_q: torch.Tensor   = log_q_flat.view(S, B)

        return samples, log_q