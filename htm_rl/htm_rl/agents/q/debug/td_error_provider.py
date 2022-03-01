# -----------------------------------------------------------------------------------------------
# © 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research Institute" (AIRI);
# Moscow Institute of Physics and Technology (National Research University). All rights reserved.
# 
# Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
# -----------------------------------------------------------------------------------------------

from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.agents.q.agent import QAgent


class TDErrorProvider(Debugger):
    agent: QAgent

    @property
    def td_error(self):
        # FIXME
        return self.agent.TD_error