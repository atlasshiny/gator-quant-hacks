import numpy as np
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any

class BitRange(BaseModel):
    name: str
    bits: int
    description: str

class StrategyChromosomeSchema(BaseModel):
    """
    Metadata schema for the 64-bit strategy chromosome.
    Enforces structural boundaries before Triton kernel allocation.
    """
    feature_flags: BitRange = Field(
        default=BitRange(name="feature_flags", bits=28, description="Boolean flags for 28 active features")
    )
    lookback_window: BitRange = Field(
        default=BitRange(name="lookback_window", bits=12, description="Gray-coded lookback length")
    )
    thresholds: BitRange = Field(
        default=BitRange(name="thresholds", bits=12, description="Z-score trigger bounds")
    )
    risk_rules: BitRange = Field(
        default=BitRange(name="risk_rules", bits=12, description="Stop-loss and sizing multiples")
    )

    @model_validator(mode="after")
    def verify_exact_64_bits(self) -> "StrategyChromosomeSchema":
        total_bits = (
            self.feature_flags.bits
            + self.lookback_window.bits
            + self.thresholds.bits
            + self.risk_rules.bits
        )
        if total_bits != 64:
            raise ValueError(f"Chromosome must be exactly 64 bits! Allocated: {total_bits} bits.")
        return self

    def compile_masks_and_shifts(self) -> Dict[str, Dict[str, Any]]:
        """
        Generates exact bitwise shift constants and hexadecimal masks for Triton/PyTorch.
        """
        fields = [
            ("feature_flags", self.feature_flags.bits),
            ("lookback_window", self.lookback_window.bits),
            ("thresholds", self.thresholds.bits),
            ("risk_rules", self.risk_rules.bits),
        ]
        
        constants = {}
        shift_accumulator = 0
        
        for name, num_bits in fields:
            unshifted_mask = (1 << num_bits) - 1
            shifted_mask = unshifted_mask << shift_accumulator
            
            constants[name] = {
                "bits": num_bits,
                "shift": shift_accumulator,
                "mask": unshifted_mask,
                "shifted_mask": shifted_mask,
                "hex_mask": hex(shifted_mask),
            }
            shift_accumulator += num_bits
            
        return constants

class StrategyChromosome:
    """
    64-Bit Chromosome Encoder / Decoder driven dynamically by StrategyChromosomeSchema.
    """
    SCHEMA = StrategyChromosomeSchema()
    CONSTANTS = SCHEMA.compile_masks_and_shifts()

    @staticmethod
    def int_to_gray(n: int) -> int:
        """Converts standard binary to Gray Code for smoother GA mutation landscapes."""
        return n ^ (n >> 1)

    @staticmethod
    def gray_to_int(g: int) -> int:
        """Converts Gray Code back to standard binary integer."""
        mask = g >> 1
        while mask != 0:
            g ^= mask
            mask >>= 1
        return g

    @classmethod
    def encode(cls, features_mask: int, lookback_raw: int, threshold_raw: int, risk_raw: int) -> np.uint64:
        """Packs strategy parameters into a single uint64 bitmask dynamically."""
        c = cls.CONSTANTS
        
        # Convert lookback parameter to Gray Code
        lookback_gray = cls.int_to_gray(lookback_raw & c["lookback_window"]["mask"])
        
        chromosome = 0
        chromosome |= (features_mask & c["feature_flags"]["mask"]) << c["feature_flags"]["shift"]
        chromosome |= (lookback_gray & c["lookback_window"]["mask"]) << c["lookback_window"]["shift"]
        chromosome |= (threshold_raw & c["thresholds"]["mask"]) << c["thresholds"]["shift"]
        chromosome |= (risk_raw & c["risk_rules"]["mask"]) << c["risk_rules"]["shift"]
        
        return np.uint64(chromosome)

    @classmethod
    def decode(cls, chromosome: int) -> dict:
        """Unpacks a uint64 bitmask integer back into human-readable parameters."""
        chromosome = int(chromosome)
        c = cls.CONSTANTS
        
        features_raw = (chromosome & c["feature_flags"]["shifted_mask"]) >> c["feature_flags"]["shift"]
        lookback_gray = (chromosome & c["lookback_window"]["shifted_mask"]) >> c["lookback_window"]["shift"]
        threshold_raw = (chromosome & c["thresholds"]["shifted_mask"]) >> c["thresholds"]["shift"]
        risk_raw      = (chromosome & c["risk_rules"]["shifted_mask"]) >> c["risk_rules"]["shift"]
        
        # Decode Gray-coded lookback window
        lookback_decoded = cls.gray_to_int(lookback_gray)
        
        return {
            "active_features_bitmask": bin(features_raw),
            "feature_flags": [(features_raw >> i) & 1 for i in range(c["feature_flags"]["bits"])],
            "lookback_window": lookback_decoded,
            "threshold_param": threshold_raw,
            "risk_param": risk_raw
        }

# Unit Test / Example Usage
if __name__ == "__main__":
    # Test Parameters
    feat_flags = 0b1010101010101010101010101010 # Selected features
    lookback = 256 # Raw lookback window
    threshold = 15 # Trigger threshold code
    risk = 7 # Risk code
    
    # Encode
    encoded_mask = StrategyChromosome.encode(feat_flags, lookback, threshold, risk)
    print(f"Encoded Chromosome (uint64): {encoded_mask} (Hex: {hex(encoded_mask)})")
    
    # Decode
    decoded_params = StrategyChromosome.decode(encoded_mask)
    print(f"Decoded Parameters: {decoded_params}")
    
    # Assert fidelity
    assert decoded_params["lookback_window"] == lookback
    assert decoded_params["threshold_param"] == threshold
    assert decoded_params["risk_param"] == risk
    print("\nEncoding/Decoding verification successful! Schema and Encoder are locked.")