#!/usr/bin/env bash
set -u

echo 'Running generated PeTTaChainer subgoal files...'

echo '============================================================'
echo 'Running modus_ponens_after_apply goal 0'
echo 'Metta: /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/modus_ponens_after_apply_goal_0.metta'
echo 'Log:   /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/modus_ponens_after_apply_goal_0.log'
petta /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/modus_ponens_after_apply_goal_0.metta > /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/modus_ponens_after_apply_goal_0.log 2>&1 || echo 'Command failed for modus_ponens_after_apply goal 0; see log.'

echo '============================================================'
echo 'Running chain_implication_after_apply_hQR goal 0'
echo 'Metta: /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/chain_implication_after_apply_hQR_goal_0.metta'
echo 'Log:   /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/chain_implication_after_apply_hQR_goal_0.log'
petta /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/chain_implication_after_apply_hQR_goal_0.metta > /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/chain_implication_after_apply_hQR_goal_0.log 2>&1 || echo 'Command failed for chain_implication_after_apply_hQR goal 0; see log.'

echo '============================================================'
echo 'Running and_commutativity_after_constructor goal 0'
echo 'Metta: /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/and_commutativity_after_constructor_goal_0.metta'
echo 'Log:   /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/and_commutativity_after_constructor_goal_0.log'
petta /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/and_commutativity_after_constructor_goal_0.metta > /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/and_commutativity_after_constructor_goal_0.log 2>&1 || echo 'Command failed for and_commutativity_after_constructor goal 0; see log.'

echo '============================================================'
echo 'Running and_commutativity_after_constructor goal 1'
echo 'Metta: /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/and_commutativity_after_constructor_goal_1.metta'
echo 'Log:   /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/and_commutativity_after_constructor_goal_1.log'
petta /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/and_commutativity_after_constructor_goal_1.metta > /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/and_commutativity_after_constructor_goal_1.log 2>&1 || echo 'Command failed for and_commutativity_after_constructor goal 1; see log.'

echo '============================================================'
echo 'Running or_elimination_after_cases goal 0'
echo 'Metta: /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/or_elimination_after_cases_goal_0.metta'
echo 'Log:   /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/or_elimination_after_cases_goal_0.log'
petta /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/or_elimination_after_cases_goal_0.metta > /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/or_elimination_after_cases_goal_0.log 2>&1 || echo 'Command failed for or_elimination_after_cases goal 0; see log.'

echo '============================================================'
echo 'Running or_elimination_after_cases goal 1'
echo 'Metta: /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/or_elimination_after_cases_goal_1.metta'
echo 'Log:   /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/or_elimination_after_cases_goal_1.log'
petta /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/or_elimination_after_cases_goal_1.metta > /home/nolawi/clone-maths/maths_ai/pln_inference/metta/tests/translator_output/or_elimination_after_cases_goal_1.log 2>&1 || echo 'Command failed for or_elimination_after_cases goal 1; see log.'

echo 'Done.'
