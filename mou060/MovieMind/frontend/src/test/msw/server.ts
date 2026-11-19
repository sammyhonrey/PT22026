import { setupServer } from "msw/node";

import { defaultHandlers } from "@/test/msw/handlers";

export const server = setupServer(...defaultHandlers);
