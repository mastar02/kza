// The optional local plan and its model must be installed together.
const privatePlan = await import('./private/house-plan.js').catch(() => null);
export const hasPrivatePlan = privatePlan !== null;
export const plan = privatePlan ?? await import('./house-plan.example.js');
