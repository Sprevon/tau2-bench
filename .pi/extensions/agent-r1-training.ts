import { createTau2TelecomExtension } from "../lib/tau2-telecom.ts";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

/**
 * Training-only adapter around the canonical Telecom extension.
 *
 * The task execution, tool registration, active-tool allowlist, and evaluator
 * remain owned by tau2-telecom. This wrapper only publishes the final evaluator
 * result as a session entry so a non-interactive Agent-R1 host can consume it.
 */
export default function agentR1TrainingExtension(pi: ExtensionAPI): void {
	const taskId = process.env.TAU2_TELECOM_TASK_ID?.trim();
	if (!taskId) {
		throw new Error("TAU2_TELECOM_TASK_ID is required by the Agent-R1 training extension");
	}

	const telecom = createTau2TelecomExtension({
		taskId,
		autoPrompt: true,
		onEvaluation: async (result) => {
			pi.appendEntry("agent-r1-evaluation", result);
		},
	});

	telecom.extension(pi);
}
