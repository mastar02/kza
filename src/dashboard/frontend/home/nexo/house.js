// Optional 3D viewer. Room readings and controls come from the current house catalog.
import * as THREE from './vendor/three.module.min.js';
import { GLTFLoader } from './vendor/three-addons/loaders/GLTFLoader.js';
import { SCAN_ROOMS, FLOOR_ROOM, MAIN_FLOOR, matchRooms } from './house-map.js';
import { hasPrivatePlan } from './house-plan.js';
import {
  DEFAULT_YAW, WALL_PATCHES, SPLIT, STAIRS, BALCONY, CUT_HEIGHT,
  HIDDEN_PREFIXES, HIDDEN_NAMES, DOOR_PREFIX, WALL_GROUP, GLASS, TINTS, LIGHT_CABINETS,
} from './house-patches.js';
import { floorTexture, planarUV } from './house-textures.js';
import { createRoomPicker, floorCentroid } from './house-picking.js';

export const MODEL_URL = '/home/nexo/private/casa.glb';
const ROTATE_STEP = Math.PI / 12; // WCAG 2.5.7: the rotate buttons cover yaw only, in fixed 15 degree steps; tilt stays on drag and the Plano/3D switch.
const DEFAULT_TILT = Math.PI * 33 / 180; // 57° above the floor.
const DEFAULT_ZOOM = 1;
const LABEL_HEIGHT = .62;
const FIT_MARGIN = 1.055;
const DRAG_THRESHOLD = 4;
const THERMAL_ACTIONS = {
  cooling: ['Enfriando', 0x71bdff], heating: ['Calefaccionando', 0xffad77],
  fan: ['Ventilando', 0x62e8cd], drying: ['Deshumidificando', 0xb5a6ff],
  preheating: ['Precalentando', 0xffad77], defrosting: ['Descongelando', 0xffad77]
};
const THERMAL_MODES = {
  cool: ['Modo frío', 0x71bdff], heat: ['Modo calor', 0xffad77], fan_only: THERMAL_ACTIONS.fan,
  dry: ['Modo seco', 0xb5a6ff], auto: ['Modo automático', 0x62e8cd], heat_cool: ['Modo frío y calor', 0x62e8cd]
};
const NEUTRAL_THERMAL = 0x8097a8;
const SELECTED_OUTLINE = 0x1f5fd1;
const degrees = value => Number.isFinite(value) ? `${value} °C` : null;

function lightReading(room, entities) {
  const lights = entities.filter(entity => entity.domain === 'light' && entity.available && entity.state === 'on');
  const primary = lights.find(entity => entity.entity_id === room.default_light) ?? lights[0];
  const kelvin = primary?.color_temp_kelvin;
  return {
    fraction: (lights.length ? Math.max(...lights.map(entity =>
      Number.isFinite(entity.brightness_pct) ? THREE.MathUtils.clamp(entity.brightness_pct, 0, 100) : 100)) : 0) / 100,
    color: kelvin < 3300 && kelvin > 0 ? 0xffc885 : kelvin >= 4800 ? 0xadd5ff : 0xf1f1ff
  };
}

export function thermalReading(room, entities) {
  const units = entities.filter(entity => entity.domain === 'climate');
  const unit = units.find(entity => entity.entity_id === room.default_climate) ?? (units.length === 1 ? units[0] : null);
  const actual = degrees(room.current_temperature ?? unit?.current_temperature);
  if (!unit?.available || ['unknown', 'unavailable'].includes(unit.state)) {
    return { actual, active: false, status: unit ? 'Estado desconocido' : '', color: NEUTRAL_THERMAL };
  }
  const effect = Object.hasOwn(THERMAL_ACTIONS, unit.hvac_action) ? THERMAL_ACTIONS[unit.hvac_action]
    : !unit.hvac_action && Object.hasOwn(THERMAL_MODES, unit.state) ? THERMAL_MODES[unit.state] : null;
  const active = unit.state !== 'off' && !!effect;
  return {
    actual, active,
    status: active ? effect[0] : unit.state === 'off' || unit.hvac_action === 'off' ? 'Apagado' : 'En pausa',
    color: active ? effect[1] : NEUTRAL_THERMAL
  };
}

function showFallback(container) {
  const fallback = document.createElement('div');
  fallback.className = 'house-fallback';
  fallback.textContent = 'No se pudo mostrar el plano. Podés usar la lista de ambientes.';
  fallback.setAttribute('role', 'status');
  container.after(fallback);
  return fallback;
}

function collectMaterials(object, materials) {
  for (const material of (Array.isArray(object.material) ? object.material : [object.material])) {
    if (material) materials.add(material);
  }
}

export function disposeObjects(roots, materials = new Set()) {
  const geometries = new Set(), textures = new Set(), images = new Set(), shadows = new Set();
  for (const root of roots) root?.traverse(object => {
    if (object.geometry) geometries.add(object.geometry);
    collectMaterials(object, materials);
    if (object.shadow) shadows.add(object.shadow);
  });
  for (const material of materials) for (const value of Object.values(material)) if (value?.isTexture) textures.add(value);
  for (const texture of textures) {
    const data = texture.source?.data ?? texture.image;
    for (const image of Array.isArray(data) ? data : [data]) {
      if (typeof image?.close === 'function') images.add(image);
    }
  }
  for (const resource of [...textures, ...geometries, ...materials, ...shadows]) resource.dispose();
  // Three.js releases GPU textures but leaves their decoded images to the caller.
  for (const image of images) image.close();
}

export function modelLoadingManager() {
  const manager = new THREE.LoadingManager();
  // GLTFLoader revokes embedded image URLs on success. The image loader reports decode/fetch failures here.
  manager.onError = url => { if (url.startsWith('blob:')) URL.revokeObjectURL(url); };
  return manager;
}

/** Render the scanned house with live state for the catalog rooms it can be matched to. */
export function createHouse({container,pins,onSelect=()=>{},onReady=()=>{},modelUrl=MODEL_URL}) {
  let renderer,scene,observer,contextStatus,rafId=0,destroyed=false,fallback=null;
  let loaded=false,glOk=true,pending=null;
  const pointers = new Map(), modelMaterials = new Set();
  let otherModelScenes = [];
  const listeners=new AbortController();
  const loading=new AbortController();
  const hasContext=()=>glOk&&!destroyed&&!renderer.getContext().isContextLost();
  const isReady=()=>loaded&&hasContext();
  function releasePointer(id) {
    const target = pointers.get(id)?.target;
    pointers.delete(id);
    if (target?.hasPointerCapture(id)) target.releasePointerCapture(id);
  }
  function destroy(){
    fallback?.remove();fallback=null;
    if(destroyed)return;
    destroyed=true;cancelAnimationFrame(rafId);observer?.disconnect();listeners.abort();loading.abort();delete container.dataset.yaw;
    contextStatus?.remove();
    for (const id of pointers.keys()) releasePointer(id);
    pins?.replaceChildren();
    const canvas=renderer?.domElement;
    disposeObjects([scene, ...otherModelScenes], modelMaterials);
    modelMaterials.clear(); otherModelScenes = [];
    renderer?.renderLists?.dispose();renderer?.dispose();renderer?.forceContextLoss();canvas?.remove();
    scene?.clear();scene=renderer=observer=null;
  }
  const noop={setView(){},zoom(){},rotate(){},yaw(){return null;},reset(){},update(){},destroy};
  function failed(){destroy();fallback=showFallback(container);onReady(false);}
  if (!hasPrivatePlan && modelUrl === MODEL_URL) {
    failed();
    return noop;
  }
  try {
    renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,preserveDrawingBuffer:true,powerPreference:'low-power'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,1.8));
  renderer.setClearColor(0x000000,0);
  renderer.shadowMap.enabled=true;
  renderer.shadowMap.autoUpdate=false;
  renderer.shadowMap.type=THREE.PCFSoftShadowMap;
  renderer.outputColorSpace=THREE.SRGBColorSpace;
  renderer.toneMapping=THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure=.95;
  renderer.localClippingEnabled=true;
  container.append(renderer.domElement);
  // The image role on the canvas container flattens descendants in the accessibility tree.
  contextStatus=document.createElement('div');
  contextStatus.setAttribute('role','status');
  container.after(contextStatus);
  scene=new THREE.Scene();
  const house=new THREE.Group();scene.add(house);
  const camera=new THREE.OrthographicCamera(-12,12,8,-8,.1,100);
  const target=new THREE.Vector3(0,.2,0);
  let yaw=DEFAULT_YAW,tilt=DEFAULT_TILT,zoom=DEFAULT_ZOOM,currentYaw=yaw,currentTilt=tilt,currentZoom=zoom;
  let width=0,height=0,dirty=true,cameraDirty=true,pinsDirty=true,rendered=false,projection=0;
  scene.add(new THREE.HemisphereLight(0xffffff,0xb9b4aa,2.35));
  const keyLight=new THREE.DirectionalLight(0xfff3e0,1.65);keyLight.position.set(5,16,6);keyLight.castShadow=true;
  Object.assign(keyLight.shadow.camera,{left:-10,right:10,top:10,bottom:-10,near:1,far:40});
  keyLight.shadow.mapSize.set(1024,1024);keyLight.shadow.bias=-.0004;keyLight.shadow.normalBias=.03;
  scene.add(keyLight);
  // One point light casts shadows (the selected room's); the rest are cheap fills.
  const focusLight=new THREE.PointLight(0xffd6a0,0,9,1.6);focusLight.castShadow=true;
  focusLight.shadow.mapSize.set(512,512);focusLight.shadow.bias=-.003;Object.assign(focusLight.shadow.camera,{near:.2,far:9});
  scene.add(focusLight);
  const cutPlane=new THREE.Plane(new THREE.Vector3(0,-1,0),CUT_HEIGHT);

  const floors=[],roomFloors={},centers={},roomLights={},outlines={};
  const floorColors = new Map();
  const hoverColor = new THREE.Color(0x7ba6ee);
  let pickRoom;
  const finishes={};
  function floorMaterial(kind){
    finishes[kind]??=floorTexture(kind);
    // The scan carries no normals (GLTFLoader flags its own materials flat for that reason), so ours must be flat too or they render black.
    return new THREE.MeshStandardMaterial({map:finishes[kind],roughness:kind==='laminado'?.7:.5,metalness:0,emissive:0x000000,emissiveIntensity:0,side:THREE.DoubleSide,flatShading:true});
  }
  function addFloor(mesh,scan){
    const kind=SCAN_ROOMS[scan].floor;
    finishes[kind]??=floorTexture(kind);
    planarUV(mesh.geometry,finishes[kind].userData.size);
    mesh.material=floorMaterial(kind);mesh.receiveShadow=true;mesh.userData.room=scan;
    floors.push(mesh);(roomFloors[scan]??=[]).push(mesh);
  }
  function box([x0,x1,y0,y1,z0,z1],material,parent){
    const geometry=new THREE.BoxGeometry(x1-x0,y1-y0,z1-z0);geometry.translate((x0+x1)/2,(y0+y1)/2,(z0+z1)/2);
    const mesh=new THREE.Mesh(geometry,material);mesh.castShadow=true;mesh.receiveShadow=true;parent.add(mesh);return mesh;
  }
  const tint=name=>TINTS.find(entry=>name.startsWith(entry.prefix));
  const isGlass=name=>GLASS.names.includes(name)||GLASS.prefixes.some(prefix=>name.startsWith(prefix));
  const glassMaterial=()=>new THREE.MeshStandardMaterial({color:0xbcd8ea,roughness:.1,metalness:0,transparent:true,opacity:.4,emissive:0xcfe6f5,emissiveIntensity:.5,side:THREE.DoubleSide,flatShading:true});

  function build(gltf){
    const root=gltf.scene;house.add(root);
    const byName={};root.traverse(object=>{if(object.name)byName[object.name]=object;});
    root.traverse(object=>{
      if(!object.isMesh)return;
      const name=object.name;
      if(HIDDEN_PREFIXES.some(prefix=>name.startsWith(prefix))||HIDDEN_NAMES.includes(name)){object.visible=false;return;}
      if(FLOOR_ROOM[name]){addFloor(object,FLOOR_ROOM[name]);return;}
      if(isGlass(name)){object.material=glassMaterial();return;}
      if(name.startsWith(DOOR_PREFIX)){object.material=new THREE.MeshStandardMaterial({color:0xf1ede2,roughness:.7,side:THREE.DoubleSide,flatShading:true});object.castShadow=true;return;}
      const material=object.material.clone();material.metalness=0;material.side=THREE.DoubleSide;
      if(object.parent?.name===WALL_GROUP)material.roughness=.95;
      else{
        material.roughness=.75;
        const entry=tint(name);
        if(entry){material.color.set(entry.color);if(entry.roughness!==undefined)material.roughness=entry.roughness;if(entry.metalness!==undefined)material.metalness=entry.metalness;}
      }
      object.material=material;object.castShadow=true;object.receiveShadow=true;
    });
    for(const name of LIGHT_CABINETS)if(byName[name]?.isMesh)byName[name].material.color.set(0xecebe6);

    const patches=new THREE.Group();house.add(patches);
    const patchMaterial=new THREE.MeshStandardMaterial({color:0xe6e6e4,roughness:.95});
    for(const wall of WALL_PATCHES)box(wall.box,patchMaterial,patches);
    if(SPLIT)box(SPLIT.box,new THREE.MeshStandardMaterial({color:0xf6f6f4,roughness:.4}),patches);
    if (BALCONY) {
      const {slab}=BALCONY;
      const slabGeometry=new THREE.BoxGeometry(slab.width,slab.thickness,slab.depth);
      slabGeometry.translate(slab.x+slab.width/2,-slab.thickness/2,slab.z);
      const slabMesh=new THREE.Mesh(slabGeometry,new THREE.MeshStandardMaterial());patches.add(slabMesh);addFloor(slabMesh,BALCONY.room);
      const railMaterial=new THREE.MeshStandardMaterial({color:0xe8dfc4,roughness:.95});
      for(const rail of BALCONY.rails){
        const mesh=new THREE.Mesh(new THREE.BoxGeometry(...rail.size),railMaterial);mesh.position.set(...rail.centre);
        mesh.castShadow=true;mesh.receiveShadow=true;patches.add(mesh);
      }
    }
    if (STAIRS) {
      const {pit}=STAIRS;
      const hole=()=>new THREE.Path([new THREE.Vector2(pit.x0,pit.z0),new THREE.Vector2(pit.x0,pit.z1),new THREE.Vector2(pit.x1,pit.z1),new THREE.Vector2(pit.x1,pit.z0)]);
      const landing=new THREE.Shape(STAIRS.landing.map(([x,z])=>new THREE.Vector2(x,z)));landing.holes.push(hole());
      const landingGeometry=new THREE.ShapeGeometry(landing);landingGeometry.rotateX(Math.PI/2);
      const landingMesh=new THREE.Mesh(landingGeometry,new THREE.MeshStandardMaterial());patches.add(landingMesh);addFloor(landingMesh,STAIRS.room);
      const depth=(pit.z1-pit.z0)/STAIRS.steps;
      for(let i=0;i<STAIRS.steps;i++){
        const tread=box([pit.x0,pit.x1,STAIRS.treadBottom,-STAIRS.rise*(i+1),pit.z1-(i+1)*depth,pit.z1-i*depth],
          new THREE.MeshStandardMaterial({color:0xa9abad,roughness:.7,emissive:0x000000,emissiveIntensity:0}),patches);
        tread.userData.room=STAIRS.room;floors.push(tread);roomFloors[STAIRS.room].push(tread);
      }
      const pitMaterial=new THREE.MeshStandardMaterial({color:0xd9d9d6,roughness:1}),wall=STAIRS.wall;
      for(const side of [[pit.x0-wall,pit.x0,pit.z0,pit.z1],[pit.x1,pit.x1+wall,pit.z0,pit.z1],[pit.x0-wall,pit.x1+wall,pit.z0-wall,pit.z0],[pit.x0-wall,pit.x1+wall,pit.z1,pit.z1+wall]])
        box([side[0],side[1],STAIRS.bottom,STAIRS.wallTop,side[2],side[3]],pitMaterial,patches);
      box(STAIRS.rail.box,new THREE.MeshStandardMaterial({color:0x1c1d20,roughness:.5}),patches);
    }

    // Use the actual floor footprint, including its recesses and the open stairwell.
    house.updateMatrixWorld(true);
    const footprint = new THREE.Box3();
    const world = new THREE.Vector3();
    for (const floor of floors) {
      footprint.expandByObject(floor);
      floorColors.set(floor, floor.material.color.clone());
    }
    for(const scan of Object.keys(SCAN_ROOMS)){
      const main=byName[MAIN_FLOOR[scan]]??roomFloors[scan]?.[0];
      if(!main)continue;
      const centre=floorCentroid(main);
      centers[scan]=centre;
      const light=new THREE.PointLight(0xffd6a0,0,9,1.6);light.position.set(centre.x,2.25,centre.z);scene.add(light);roomLights[scan]=light;
      // Attach the outline to the first visible floor for this room.
      const drawn=roomFloors[scan]?.[0];
      if(!drawn)continue;
      const outline=new THREE.LineSegments(new THREE.EdgesGeometry(drawn.geometry),new THREE.LineBasicMaterial({color:SELECTED_OUTLINE,depthTest:false,transparent:true}));
      outline.renderOrder=2;outline.visible=false;drawn.add(outline);outlines[scan]=outline;
    }
    house.traverse(object=>{
      if(!object.isMesh)return;
      for(const material of (Array.isArray(object.material)?object.material:[object.material])){material.clippingPlanes=[cutPlane];material.clipShadows=true;}
    });
    house.updateMatrixWorld(true);
    footprint.getCenter(target);
    target.y = .2;
    const pickable = [];
    house.traverseVisible(object => {
      if (!object.isMesh) return;
      const positions = object.geometry.getAttribute('position');
      const points = [];
      for (let index = 0; index < positions.count; index++) {
        world.fromBufferAttribute(positions, index).applyMatrix4(object.matrixWorld);
        points.push(world.clone());
      }
      if (points.every(point => point.y > CUT_HEIGHT)) return;
      pickable.push(object);
      for (const point of points) {
        // Frame the cutaway at wall height and above the bottom of the stairwell.
        point.y = THREE.MathUtils.clamp(point.y, -.11, CUT_HEIGHT);
        framePoints.push(point);
      }
    });
    pickRoom = createRoomPicker(pickable, floors, cutPlane);
  }

  const pinPosition=new THREE.Vector3();
  const gap=4;
  const offsets=[];
  for (const y of [0,-16,16,-34,34,-54,54,-78,78,-110,110,-145,145]) {
    for (const x of [0,-24,24,-52,52,-84,84,-124,124]) offsets.push([x,y]);
  }
  offsets.sort(([x1,y1],[x2,y2]) => Math.hypot(x1,y1) - Math.hypot(x2,y2));
  let bindings=[],signature,hovered;
  const pinRooms = new WeakMap();
  let byZone = new Map();
  function projectPins() {
    const occupied = [];
    const anchors = Object.entries(centers).map(([zone, centre]) => {
      pinPosition.copy(centre).project(camera);
      return { zone, x: (pinPosition.x + 1) * width / 2, y: (1 - pinPosition.y) * height / 2 };
    });
    const targets = bindings.map(({ pin, zone }) => {
      pinPosition.copy(centers[zone]); pinPosition.y = LABEL_HEIGHT; pinPosition.project(camera);
      const { width: w, height: h } = pin.getBoundingClientRect();
      return { pin, zone, w, h, x: (pinPosition.x + 1) * width / 2, y: (1 - pinPosition.y) * height / 2 };
    });
    const overlaps = ({ left, top, right, bottom }, zone) => occupied.some(rect =>
      left < rect.right+gap && right > rect.left-gap && top < rect.bottom+gap && bottom > rect.top-gap)
      || anchors.some(({ zone: other, x, y }) =>
        zone !== other && x >= left && x <= right && y >= top && y <= bottom);
    // Let pins with competing anchors choose a nearby spot before isolated pins.
    const neighbours = target => targets.filter(other => target !== other
      && Math.abs(target.x - other.x) < (target.w + other.w) / 2 + gap
      && Math.abs(target.y - other.y) < (target.h + other.h) / 2 + gap).length;
    const priorities = new Map(targets.map(target => [target, neighbours(target)]));
    targets.sort((a, b) => priorities.get(b) - priorities.get(a));
    for (const { pin, zone, x, y, w, h } of targets) {
      let position;
      for (const [dx,dy] of offsets) {
        const left = THREE.MathUtils.clamp(x+dx-w/2, gap, width-w-gap);
        const top = THREE.MathUtils.clamp(y+dy-h/2, gap, height-h-gap);
        const candidate = {left,top,right:left+w,bottom:top+h};
        if (!overlaps(candidate, zone)) {
          position = candidate; break;
        }
      }
      if (!position) {
        // Larger text can leave slots between the fixed offsets. Also try packing against occupied edges.
        const lefts = [gap, ...occupied.map(rect => rect.right+gap)];
        const tops = [gap, ...occupied.map(rect => rect.bottom+gap)];
        let nearest = Infinity;
        for (const left of lefts) for (const top of tops) {
          const candidate = {left,top,right:left+w,bottom:top+h};
          const distance = Math.hypot(left+w/2-x, top+h/2-y);
          if (candidate.right <= width-gap && candidate.bottom <= height-gap && distance < nearest && !overlaps(candidate, zone)) {
            position = candidate;
            nearest = distance;
          }
        }
      }
      pin.style.visibility = position ? '' : 'hidden';
      if (position) {
        occupied.push(position);
        pin.style.translate = `${position.left}px ${position.top}px`;
      }
    }
  }

  function setHover(scan) {
    if (scan === hovered) return;
    for (const floor of roomFloors[hovered] ?? []) floor.material.color.copy(floorColors.get(floor));
    byZone.get(hovered)?.pin.removeAttribute('data-room-hover');
    hovered = scan;
    for (const floor of roomFloors[scan] ?? []) floor.material.color.lerp(hoverColor, .26);
    byZone.get(scan)?.pin.setAttribute('data-room-hover', '');
    canvas.toggleAttribute('data-room-hover', !!scan);
    dirty = true;
    scheduleFrame();
  }

  function apply({catalog,layer,selected}) {
    const entities = new Map((catalog?.entities ?? []).map(entity => [entity.entity_id, entity]));
    const climate = layer === 'climate';
    const matched = matchRooms(catalog?.rooms ?? []);
    const rooms = (catalog?.rooms ?? []).filter(room => centers[matched.get(room.room_id)]).map(room => {
      const devices = room.entity_ids.map(id => entities.get(id)).filter(Boolean);
      return {
        id:room.room_id, name:room.name, zone:matched.get(room.room_id), selected:room.room_id === selected,
        reading:climate ? thermalReading(room, devices) : lightReading(room, devices)
      };
    });
    const nextSignature = JSON.stringify([climate, rooms]);
    if (nextSignature === signature) return rooms.length;
    signature = nextSignature;
    const previous = new Map(bindings.map(binding => [binding.id, binding]));
    bindings = rooms.map(room => {
      const old = previous.get(room.id);
      let pin = old?.pin;
      if (old?.zone !== room.zone) pinsDirty = true;
      previous.delete(room.id);
      if (!pin) {
        pin = document.createElement('button');
        pin.type = 'button'; pin.tabIndex = -1; pin.className = 'room-pin';
        pinRooms.set(pin, room.id);
        const label = document.createElement('span');
        label.className = 'tag';
        pin.append(label);
        if (pins) {
          pins.append(pin);
          observer.observe(pin, {box:'border-box'});
        }
      }
      pin.setAttribute('aria-pressed', String(room.selected));
      if (pin.firstChild.textContent !== room.name) pin.firstChild.textContent = room.name;
      pin.toggleAttribute('data-room-hover', room.zone === hovered);
      return { ...room, pin };
    });
    for (const { pin } of previous.values()) {
      for (const [id, pointer] of pointers) if (pointer.target === pin) releasePointer(id);
      observer.unobserve(pin);
      pin.remove();
      pinsDirty = true;
    }
    byZone = new Map(bindings.map(room => [room.zone, room]));
    if (hovered && !byZone.has(hovered)) setHover(undefined);
    let focus=null;
    for (const scan of Object.keys(SCAN_ROOMS)) {
      const room = byZone.get(scan), reading = room?.reading;
      const fraction = climate ? 0 : reading?.fraction ?? 0;
      const color = reading?.color ?? 0xffffff;
      const light = roomLights[scan];
      if (light) { light.color.set(color); light.intensity = 0; }
      if (fraction > 0 && room.selected && light) focus = { light, fraction, color };
      else if (fraction > 0 && light) light.intensity = 16*fraction+3;
      const thermal = climate && !!room;
      for (const floor of roomFloors[scan] ?? []) {
        floor.material.emissive.set(color);
        floor.material.emissiveIntensity = thermal ? (reading.active ? .15 : .03) : fraction*.2;
      }
      const outline = outlines[scan];
      if (outline) {
        outline.visible = !!room && (climate || room.selected);
        outline.material.color.set(room?.selected ? SELECTED_OUTLINE : color);
      }
    }
    focusLight.intensity = focus ? 16*focus.fraction+3 : 0;
    if (focus) { focusLight.color.set(focus.color); focusLight.position.copy(focus.light.position); }
    renderer.shadowMap.needsUpdate = true;
    dirty = true;
    scheduleFrame();
    return rooms.length;
  }
  function update(args) {
    if (destroyed) return;
    if (!loaded) { pending = args; return; }
    const linked = apply(args);
    return glOk ? linked : undefined;
  }
  pins?.addEventListener('click', event => {
    const pin = event.target.closest('.room-pin');
    if (isReady() && pinRooms.has(pin) && event.detail === 0) onSelect(pinRooms.get(pin));
  }, {signal:listeners.signal});
  function resize(){
    if(destroyed)return;
    width=container.clientWidth;height=container.clientHeight;
    rendered=false;
    if(!width||!height)return;
    renderer.setSize(width,height,false);cameraDirty=true;
  }
  observer=new ResizeObserver(entries=>{
    if(entries.some(entry=>entry.target===container))resize();
    pinsDirty=true;
    scheduleFrame();
  });
  observer.observe(container);resize();
  const framePoints=[];
  const cameraPoint=new THREE.Vector3();
  const motionQuery=matchMedia('(prefers-reduced-motion: reduce)');
  const reducedMotion=motionQuery.matches;
  const publishYaw=()=>{container.dataset.yaw=yaw.toFixed(4);};
  publishYaw();
  function cameraUpdate(){
    const a=width/height;
    camera.position.set(target.x+26*Math.sin(currentTilt)*Math.sin(currentYaw),26*Math.cos(currentTilt),target.z+26*Math.sin(currentTilt)*Math.cos(currentYaw));camera.lookAt(target);camera.updateMatrixWorld();
    let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
    for(const corner of framePoints){cameraPoint.copy(corner).applyMatrix4(camera.matrixWorldInverse);minX=Math.min(minX,cameraPoint.x);maxX=Math.max(maxX,cameraPoint.x);minY=Math.min(minY,cameraPoint.y);maxY=Math.max(maxY,cameraPoint.y);}
    const span=Math.max(maxY-minY,(maxX-minX)/a)*FIT_MARGIN/currentZoom;
    const cx=(minX+maxX)/2,cy=(minY+maxY)/2;
    camera.left=cx-span*a/2;camera.right=cx+span*a/2;camera.top=cy+span/2;camera.bottom=cy-span/2;camera.updateProjectionMatrix();
    projection++;
  }
  let lastTime=0;
  function checkRenderContext() {
    if (hasContext()) return true;
    pauseRendering();
    return false;
  }
  function scheduleFrame() {
    if (!rafId && loaded && checkRenderContext() && width && height && !document.hidden) rafId = requestAnimationFrame(frame);
  }
  function frame(time){
    rafId = 0;
    if(!checkRenderContext()||!width||!height||document.hidden)return;
    if(time-lastTime<30){scheduleFrame();return;}
    lastTime=time;
    const diff=Math.abs(currentYaw-yaw)+Math.abs(currentTilt-tilt)+Math.abs(currentZoom-zoom);
    if(diff>.0001){
      const ease=reducedMotion||motionQuery.matches?1:.13;
      currentYaw+=(yaw-currentYaw)*ease;currentTilt+=(tilt-currentTilt)*ease;currentZoom+=(zoom-currentZoom)*ease;
      cameraDirty=true;
      if(ease<1)scheduleFrame();
    }
    if(cameraDirty){cameraUpdate();cameraDirty=false;dirty=true;pinsDirty=true;}
    if(dirty){
      if (!checkRenderContext()) return;
      try {
        renderer.render(scene,camera);
      } catch (error) {
        if (!renderer.getContext().isContextLost()) throw error;
        pauseRendering();
        return;
      }
      if (!checkRenderContext()) return;
      dirty=false;rendered=true;
    }
    if(pinsDirty){projectPins();pinsDirty=false;}
  }
  const canvas=renderer.domElement;
  const stage = canvas.closest('.map-viewport') ?? container;
  const listen = (element, event, callback, options) =>
    element.addEventListener(event, callback, { ...options, signal: listeners.signal });
  let suppressSelection = false, pinching = false;
  function geometryAt(event) {
    if (!isReady() || !rendered || !byZone.size) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    if (x < 0 || x >= rect.width || y < 0 || y >= rect.height) return;
    return pickRoom(x, y, rect.width, rect.height, camera, projection, byZone);
  }
  function roomAt(event, pin) {
    const pinRoom = pin && bindings.find(({ pin: button }) => button === pin);
    if (pinRoom) {
      const { left, right, top, bottom } = pin.firstElementChild.getBoundingClientRect();
      if (event.clientX >= left && event.clientX <= right && event.clientY >= top && event.clientY <= bottom) return pinRoom;
    }
    // Transparent padding enlarges the pin without stealing another room's visible surface.
    const hit = geometryAt(event);
    return hit?.surface || !pinRoom ? byZone.get(hit?.scan) : pinRoom;
  }
  function pinAt(target) {
    const pin = target.closest('.room-pin');
    return pin && pins?.contains(pin) ? pin : null;
  }
  listen(stage, 'pointerdown', event => {
    if (!isReady() || event.button !== 0) return;
    const target = pinAt(event.target) ?? (event.target === canvas ? canvas : null);
    if (!target) return;
    if (!pointers.size) {
      suppressSelection = false;
      pinching = false;
    } else {
      suppressSelection = true;
      pinching = true;
    }
    const { clientX: x, clientY: y } = event;
    pointers.set(event.pointerId, { target, x, y, startX: x, startY: y, travelled: 0 });
    target.setPointerCapture(event.pointerId);
    setHover(undefined);
  });
  function movePointer(event, pointer) {
    const { clientX: x, clientY: y } = event;
    const distance = Math.hypot(x - pointer.x, y - pointer.y);
    if (!distance) return;
    if (pinching) {
      if (pointers.size === 2) {
        const other = [...pointers.values()].find(item => item !== pointer);
        const before = Math.hypot(pointer.x - other.x, pointer.y - other.y);
        const after = Math.hypot(x - other.x, y - other.y);
        if (before > 0) {
          zoom = THREE.MathUtils.clamp(zoom * after / before, .65, 1.65);
          scheduleFrame();
        }
      }
    } else {
      const before = pointer.travelled;
      pointer.travelled += distance;
      if (pointer.travelled > DRAG_THRESHOLD) {
        suppressSelection = true;
        yaw -= (x - (before > DRAG_THRESHOLD ? pointer.x : pointer.startX)) * .006;
        tilt = THREE.MathUtils.clamp(tilt + (y - (before > DRAG_THRESHOLD ? pointer.y : pointer.startY)) * .003, .12, 1.18);
        publishYaw();
        scheduleFrame();
      }
    }
    Object.assign(pointer, { x, y });
  }
  listen(stage, 'pointermove', event => {
    const pointer = pointers.get(event.pointerId);
    if (pointer) movePointer(event, pointer);
    else if (event.pointerType === 'mouse' && !pointers.size) {
      setHover(roomAt(event, pinAt(event.target))?.zone);
    }
  });
  listen(stage, 'pointerleave', () => setHover(undefined));
  listen(stage, 'pointerup', event => {
    const pointer = pointers.get(event.pointerId);
    if (!pointer) return;
    movePointer(event, pointer);
    releasePointer(event.pointerId);
    if (suppressSelection || !isReady()) return;
    const room = roomAt(event, pinAt(pointer.target));
    if (room) onSelect(room.id);
  });
  for (const type of ['pointercancel', 'lostpointercapture']) {
    listen(stage, type, event => {
      if (!pointers.has(event.pointerId)) return;
      suppressSelection = true;
      releasePointer(event.pointerId);
    });
  }
  listen(stage, 'wheel', event => {
    if (pointers.size) suppressSelection = true;
    setHover(undefined);
    if (event.ctrlKey || event.metaKey) {
      event.preventDefault();
      zoom = THREE.MathUtils.clamp(zoom - event.deltaY * .002, .65, 1.65);
      scheduleFrame();
    }
  }, {passive:false});
  function pauseRendering() {
    if (!glOk || destroyed) return;
    glOk=false;rendered=false;
    cancelAnimationFrame(rafId);rafId=0;
    setHover(undefined);
    for (const id of pointers.keys()) releasePointer(id);
    if(pins)pins.hidden=true;
    contextStatus.className='house-fallback';
    contextStatus.textContent='El plano 3D se pausó. Volviendo a conectar…';
    onReady(false);
  }
  listen(canvas, 'webglcontextlost',event=>{
    event.preventDefault();
    pauseRendering();
  });
  listen(canvas, 'webglcontextrestored',()=>{
    if (renderer.getContext().isContextLost()) return;
    glOk=true;dirty=true;
    renderer.shadowMap.autoUpdate=false;
    renderer.shadowMap.needsUpdate=true;
    if(pins)pins.hidden=false;
    contextStatus.textContent='';
    contextStatus.className='';
    if(isReady()){scheduleFrame();onReady(true);}
  });
  listen(document, 'visibilitychange', scheduleFrame);
  if (pins) pins.hidden = false;
  (async()=>{
    try{
      const response=await fetch(modelUrl,{credentials:'same-origin',signal:loading.signal});
      if(!response.ok)throw new Error(`model_${response.status}`);
      const buffer=await response.arrayBuffer();
      const gltf=await new Promise((resolve,reject)=>new GLTFLoader(modelLoadingManager()).parse(buffer,'',resolve,reject));
      if(destroyed){disposeObjects(gltf.scenes);return;}
      otherModelScenes = gltf.scenes.filter(root => root !== gltf.scene);
      // build replaces some materials, so keep the originals until final teardown.
      gltf.scene.traverse(object => collectMaterials(object, modelMaterials));
      build(gltf);
      loaded=true;cameraDirty=true;
      renderer.shadowMap.needsUpdate=true;
      if(pending){const args=pending;pending=null;apply(args);}
      scheduleFrame();
      if(isReady())onReady(true);
    }catch(error){
      if(destroyed||error?.name==='AbortError')return;
      failed();
    }
  })();
  return {
    setView(view){if(destroyed)return;tilt=view==='top'?.025:DEFAULT_TILT;yaw=DEFAULT_YAW;publishYaw();zoom=DEFAULT_ZOOM;setHover(undefined);scheduleFrame();},
    /** Turn the model by one fixed step (+1 right, -1 left); under reduced motion the step lands without easing. */
    rotate(direction){if(destroyed)return;yaw+=(direction<0?1:-1)*ROTATE_STEP;if(motionQuery.matches){currentYaw=yaw;cameraDirty=true;}publishYaw();setHover(undefined);scheduleFrame();},
    yaw(){return destroyed?null:yaw;},
    zoom(delta){if(destroyed)return;zoom=THREE.MathUtils.clamp(zoom+delta,.65,1.65);setHover(undefined);scheduleFrame();},
    reset(){if(destroyed)return;yaw=DEFAULT_YAW;publishYaw();tilt=DEFAULT_TILT;zoom=DEFAULT_ZOOM;setHover(undefined);scheduleFrame();},update,destroy
  };
  } catch {
    failed();
    return noop;
  }
}
