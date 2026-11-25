// A fictional rectangular layout with three adjacent spaces. This data is an
// example of the plan format for development and tests.
export const SCAN_ROOMS = {
  lectura: { label: 'Sala de lectura', floor: 'laminado', aliases: ['lectura', 'sala de lectura', 'reading'] },
  taller: { label: 'Taller', floor: 'porcelanato', aliases: ['taller', 'workshop'] },
  musica: { label: 'Sala de música', floor: 'laminado', aliases: ['música', 'sala de música', 'music'] },
};

export const FLOOR_ROOM = { ExampleA: 'lectura', ExampleB: 'taller', ExampleC: 'musica' };
export const MAIN_FLOOR = { lectura: 'ExampleA', taller: 'ExampleB', musica: 'ExampleC' };

export const DEFAULT_YAW = 0;
export const CUT_HEIGHT = 1;
export const WALL_PATCHES = [
  { box: [0, 9, 0, 2, 0, .1] },
  { box: [0, 9, 0, 2, 4, 4.1] },
  { box: [0, .1, 0, 2, 0, 4] },
  { box: [9, 9.1, 0, 2, 0, 4] },
  { box: [3, 3.1, 0, 2, 0, 4] },
  { box: [6, 6.1, 0, 2, 0, 4] },
];
export const SPLIT = null;
export const STAIRS = null;
export const BALCONY = null;
export const HIDDEN_PREFIXES = [];
export const HIDDEN_NAMES = [];
export const DOOR_PREFIX = 'ExampleDoor';
export const WALL_GROUP = 'ExampleWalls';
export const GLASS = { prefixes: [], names: [] };
export const TINTS = [];
export const LIGHT_CABINETS = [];
