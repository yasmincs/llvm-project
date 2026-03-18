//===- InferAlignment.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Infer alignment for load, stores and other memory operations based on
// trailing zero known bits information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/InferAlignment.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include <optional>

using namespace llvm;
using namespace llvm::PatternMatch;

static bool tryToImproveAlign(
    const DataLayout &DL, Instruction *I,
    function_ref<Align(Value *PtrOp, Align OldAlign, Align PrefAlign)> Fn) {

  if (auto *PtrOp = getLoadStorePointerOperand(I)) {
    Align OldAlign = getLoadStoreAlignment(I);
    Align PrefAlign = DL.getPrefTypeAlign(getLoadStoreType(I));

    Align NewAlign = Fn(PtrOp, OldAlign, PrefAlign);
    if (NewAlign > OldAlign) {
      setLoadStoreAlignment(I, NewAlign);
      return true;
    }
  }

  Value *PtrOp;
  const APInt *Const;
  if (match(I, m_And(m_PtrToIntOrAddr(m_Value(PtrOp)), m_APInt(Const)))) {
    Align ActualAlign = Fn(PtrOp, Align(1), Align(1));
    if (Const->ult(ActualAlign.value())) {
      I->replaceAllUsesWith(Constant::getNullValue(I->getType()));
      return true;
    }
    if (Const->uge(
            APInt::getBitsSetFrom(Const->getBitWidth(), Log2(ActualAlign)))) {
      I->replaceAllUsesWith(I->getOperand(0));
      return true;
    }
  }
  if (match(I, m_Trunc(m_PtrToIntOrAddr(m_Value(PtrOp))))) {
    Align ActualAlign = Fn(PtrOp, Align(1), Align(1));
    if (Log2(ActualAlign) >= I->getType()->getScalarSizeInBits()) {
      I->replaceAllUsesWith(Constant::getNullValue(I->getType()));
      return true;
    }
  }

  IntrinsicInst *II = dyn_cast<IntrinsicInst>(I);
  if (!II)
    return false;

  // TODO: Handle more memory intrinsics.
  switch (II->getIntrinsicID()) {
  case Intrinsic::masked_load:
  case Intrinsic::masked_store: {
    unsigned PtrOpIdx = II->getIntrinsicID() == Intrinsic::masked_load ? 0 : 1;
    Value *PtrOp = II->getArgOperand(PtrOpIdx);
    Type *Type = II->getIntrinsicID() == Intrinsic::masked_load
                     ? II->getType()
                     : II->getArgOperand(0)->getType();

    Align OldAlign = II->getParamAlign(PtrOpIdx).valueOrOne();
    Align PrefAlign = DL.getPrefTypeAlign(Type);
    Align NewAlign = Fn(PtrOp, OldAlign, PrefAlign);
    if (NewAlign <= OldAlign)
      return false;

    II->addParamAttr(PtrOpIdx,
                     Attribute::getWithAlignment(II->getContext(), NewAlign));
    return true;
  }
  default:
    return false;
  }
}

bool inferAlignment(Function &F, AssumptionCache &AC, DominatorTree &DT) {
  const DataLayout &DL = F.getDataLayout();
  bool Changed = false;

  // Enforce preferred type alignment if possible. We do this as a separate
  // pass first, because it may improve the alignments we infer below.
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      Changed |= tryToImproveAlign(
          DL, &I, [&](Value *PtrOp, Align OldAlign, Align PrefAlign) {
            if (PrefAlign > OldAlign)
              return std::max(OldAlign,
                              tryEnforceAlignment(PtrOp, PrefAlign, DL));
            return OldAlign;
          });
    }
  }

  // Compute alignment from known bits.
  auto InferFromKnownBits = [&](Instruction &I, Value *PtrOp) {
    KnownBits Known = computeKnownBits(PtrOp, DL, &AC, &I, &DT);
    unsigned TrailZ =
        std::min(Known.countMinTrailingZeros(), +Value::MaxAlignmentExponent);
    return Align(1ull << std::min(Known.getBitWidth() - 1, TrailZ));
  };

  // Helper function to quickly compute alignment from common patterns
  std::function<std::optional<Align>(Value *)> getIndexAlignmentFromPattern =
      [&](Value *Idx) -> std::optional<Align> {
    // Pattern: shl X, N -> alignment of 2^N
    const APInt *ShiftAmt;
    if (match(Idx, m_Shl(m_Value(), m_APInt(ShiftAmt)))) {
      uint64_t Shift = ShiftAmt->getZExtValue();
      if (Shift > 0 && Shift <= 63)
        return Align(1ull << Shift);
    }

    // Pattern: mul X, C where C is a power of 2 -> alignment of C
    const APInt *MulC;
    if (match(Idx, m_Mul(m_Value(), m_APInt(MulC))) ||
        match(Idx, m_Mul(m_APInt(MulC), m_Value()))) {
      uint64_t Val = MulC->getZExtValue();
      if (isPowerOf2_64(Val))
        return Align(Val);
    }

    // Pattern: add X, C -> GCD of X's alignment and C
    Value *AddOp;
    const APInt *AddC;
    if (match(Idx, m_Add(m_Value(AddOp), m_APInt(AddC))) ||
        match(Idx, m_Add(m_APInt(AddC), m_Value(AddOp)))) {
      if (auto XAlign = getIndexAlignmentFromPattern(AddOp))
        return commonAlignment(*XAlign, AddC->getZExtValue());
    }

    // Pattern: sub X, C -> GCD of X's alignment and C
    if (match(Idx, m_Sub(m_Value(AddOp), m_APInt(AddC)))) {
      if (auto XAlign = getIndexAlignmentFromPattern(AddOp))
        return commonAlignment(*XAlign, AddC->getZExtValue());
    }

    // Pattern: sext/zext - extensions preserve alignment
    Value *CastSrc;
    if (match(Idx, m_SExt(m_Value(CastSrc))) ||
        match(Idx, m_ZExt(m_Value(CastSrc))))
      return getIndexAlignmentFromPattern(CastSrc);

    return std::nullopt;
  };

  // Helper function to compute variable offset alignment and base pointer
  // If ConstOffset > 0, the effective offset alignment is limited by the
  // constant offset
  auto computeVariableOffsetAlignment =
      [&](Value *Ptr, Instruction *I,
          uint64_t ConstOffset = 0) -> std::pair<Value *, Align> {
    Align VarOffsetAlign = Align(1);
    Value *VarBasePtr = Ptr;

    if (auto *GEP = dyn_cast<GEPOperator>(VarBasePtr)) {
      // We can only handle GEPs with a single index
      if (GEP->getNumIndices() == 1) {
        Value *Idx = GEP->idx_begin()->get();
        Align IndexAlign(1);

        if (auto PatternAlign = getIndexAlignmentFromPattern(Idx)) {
          IndexAlign = *PatternAlign;
        }
        // If pattern matching fails, IndexAlign remains 1 (no alignment from
        // variable offset)
        Type *EltTy = GEP->getSourceElementType();
        TypeSize EltSizeType = DL.getTypeAllocSize(EltTy);

        // If we encounter a scalable type, we can't compute alignment
        if (!EltSizeType.isScalable()) {
          uint64_t EltSize = EltSizeType.getFixedValue();

          // Compute offset alignment: multiply index alignment by element size,
          // then take the greatest power of 2 that divides the product
          uint64_t Product = IndexAlign.value() * EltSize;
          uint64_t ProductAlignValue = Product > 0
                                           ? (Product & (~Product + 1))
                                           : 1; // Extract lowest set bit
          VarOffsetAlign = Align(ProductAlignValue);
        }

        VarBasePtr = GEP->getPointerOperand();
      }
    }
    VarBasePtr = VarBasePtr->stripPointerCasts();

    // If we have a constant offset, the effective alignment is the GCD of both
    if (ConstOffset > 0) {
      VarOffsetAlign = commonAlignment(VarOffsetAlign, ConstOffset);
    }

    return {VarBasePtr, VarOffsetAlign};
  };

  // Propagate alignment between loads and stores that originate from the
  // same base pointer.
  DenseMap<Value *, Align> BestBasePointerAligns;

  // Compute final alignment from a base pointer and offset.
  // UseConstOffset: if true, use ConstOffset; if false, use VarOffsetAlign
  auto computeFinalAlign = [&](Value *BasePtr, Align FallbackAlign,
                               bool UseConstOffset, uint64_t ConstOffset,
                               Align VarOffsetAlign) -> Align {
    Align StoredBaseAlign = Align(1);
    if (auto It = BestBasePointerAligns.find(BasePtr);
        It != BestBasePointerAligns.end()) {
      StoredBaseAlign = It->second;
    }

    Align BaseAlign =
        StoredBaseAlign > Align(1) ? StoredBaseAlign : FallbackAlign;

    // Apply offset alignment (either constant or variable)
    if (UseConstOffset) {
      return commonAlignment(BaseAlign, ConstOffset);
    } else {
      return commonAlignment(BaseAlign, VarOffsetAlign.value());
    }
  };

  auto InferFromBasePointer = [&](Value *PtrOp, Align LoadStoreAlign,
                                  Instruction *I) {
    // Handle constant offsets
    APInt OffsetFromBase(DL.getIndexTypeSizeInBits(PtrOp->getType()), 0);
    Value *ConstBasePtr =
        PtrOp->stripAndAccumulateConstantOffsets(DL, OffsetFromBase, true);
    uint64_t ConstOffsetVal = OffsetFromBase.abs().getLimitedValue();

    // Derive the base pointer alignment from the load/store alignment
    // and the offset from the base pointer.
    Align BasePointerAlign = commonAlignment(LoadStoreAlign, ConstOffsetVal);

    auto [It, Inserted] =
        BestBasePointerAligns.try_emplace(ConstBasePtr, BasePointerAlign);
    if (!Inserted) {
      // If the stored base pointer alignment is better than the
      // base pointer alignment we derived, we may be able to use it
      // to improve the load/store alignment. If not, store the
      // improved base pointer alignment for future iterations.
      if (It->second < BasePointerAlign) {
        It->second = BasePointerAlign;
      }
    }

    // Handle variable offsets (constant offset is handled inside the function)
    auto [VarBasePtr, VarOffsetAlign] =
        computeVariableOffsetAlignment(ConstBasePtr, I, ConstOffsetVal);

    // Compute final alignment for constant method
    Align ConstFinalAlign = computeFinalAlign(ConstBasePtr, LoadStoreAlign,
                                              true, ConstOffsetVal, Align(1));

    // Compute final alignment for variable method
    Align VarFinalAlign =
        computeFinalAlign(VarBasePtr, LoadStoreAlign, false, 0, VarOffsetAlign);

    // Return the larger of the two final alignments
    return std::max(ConstFinalAlign, VarFinalAlign);
  };

  for (BasicBlock &BB : F) {
    // We need to reset the map for each block because alignment information
    // can only be propagated from instruction A to B if A dominates B.
    // This is because control flow (and exception throwing) could be dependent
    // on the address (and its alignment) at runtime. Some sort of dominator
    // tree approach could be better, but doing a simple forward pass through a
    // single basic block is correct too.
    BestBasePointerAligns.clear();

    // First pass: Process assumes to populate base pointer alignment map
    for (Instruction &I : BB) {
      auto *Assume = dyn_cast<IntrinsicInst>(&I);
      if (!Assume || Assume->getIntrinsicID() != Intrinsic::assume)
        continue;

      // Extract alignment information from assume operand bundles
      for (unsigned Idx = 0; Idx < Assume->getNumOperandBundles(); ++Idx) {
        OperandBundleUse OB = Assume->getOperandBundleAt(Idx);
        if (OB.getTagID() != LLVMContext::OB_align)
          continue;

        if (OB.Inputs.size() < 2)
          continue;

        Value *AAPtr = OB.Inputs[0].get();

        // Get alignment value
        if (auto *AlignVal = dyn_cast<ConstantInt>(OB.Inputs[1].get())) {
          uint64_t AlignValue = AlignVal->getZExtValue();
          if (!isPowerOf2_64(AlignValue))
            continue;
          Align AssumedAlign(AlignValue);

          // Handle constant offsets
          APInt OffsetFromBase(DL.getIndexTypeSizeInBits(AAPtr->getType()), 0);
          Value *ConstBasePtr = AAPtr->stripAndAccumulateConstantOffsets(
              DL, OffsetFromBase, true);
          uint64_t ConstOffsetVal = OffsetFromBase.abs().getLimitedValue();

          // Handle variable offsets (constant offset is handled inside the
          // function)
          auto [VarBasePtr, VarOffsetAlign] =
              computeVariableOffsetAlignment(ConstBasePtr, &I, ConstOffsetVal);

          // Compute base alignments
          Align ConstBaseAlign = computeFinalAlign(
              ConstBasePtr, AssumedAlign, true, ConstOffsetVal, Align(1));
          Align VarBaseAlign = computeFinalAlign(VarBasePtr, AssumedAlign,
                                                 false, 0, VarOffsetAlign);

          // Store alignment for both base pointers if they're different
          if (ConstBasePtr != VarBasePtr) {
            // Store alignment for constant base
            auto [ConstIt, ConstInserted] =
                BestBasePointerAligns.try_emplace(ConstBasePtr, ConstBaseAlign);
            if (!ConstInserted && ConstBaseAlign > ConstIt->second) {
              ConstIt->second = ConstBaseAlign;
            }

            // Store alignment for variable base
            auto [VarIt, VarInserted] =
                BestBasePointerAligns.try_emplace(VarBasePtr, VarBaseAlign);
            if (!VarInserted && VarBaseAlign > VarIt->second) {
              VarIt->second = VarBaseAlign;
            }
          } else {
            // Same base pointer, choose the better alignment
            Align BestBaseAlign = std::max(ConstBaseAlign, VarBaseAlign);
            auto [It, Inserted] =
                BestBasePointerAligns.try_emplace(ConstBasePtr, BestBaseAlign);
            if (!Inserted && BestBaseAlign > It->second) {
              It->second = BestBaseAlign;
            }
          }
        }
      }
    }

    // Process loads/stores and use the alignment map
    for (Instruction &I : BB) {
      Changed |= tryToImproveAlign(
          DL, &I, [&](Value *PtrOp, Align OldAlign, Align PrefAlign) {
            return std::max(InferFromKnownBits(I, PtrOp),
                            InferFromBasePointer(PtrOp, OldAlign, &I));
          });
    }
  }

  return Changed;
}

PreservedAnalyses InferAlignmentPass::run(Function &F,
                                          FunctionAnalysisManager &AM) {
  AssumptionCache &AC = AM.getResult<AssumptionAnalysis>(F);
  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  inferAlignment(F, AC, DT);
  // Changes to alignment shouldn't invalidated analyses.
  return PreservedAnalyses::all();
}
