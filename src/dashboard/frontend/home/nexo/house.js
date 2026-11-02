// Geometry adapted from approved NEXO reference d686fcde; illustrative only.
import * as THREE from './vendor/three.module.min.js';
const ROOM_LAYOUT = {
  living:{x:-2.6,z:2.05,w:6.5,d:3.75,pin:[-2.9,1.25,2.4]},
  kitchen:{x:-4.05,z:-2.05,w:3.7,d:3.7,pin:[-4,1.25,-2.0]},
  bedroom:{x:3.55,z:-2.05,w:4.6,d:3.7,pin:[3.5,1.2,-2.2]},
  office:{x:2.55,z:2.1,w:2.6,d:3.55,pin:[2.25,1.2,2.1]},
  bathroom:{x:5.05,z:2.1,w:1.95,d:3.55,pin:[5.15,1.3,2.05]},
  garden:{x:7.7,z:0,w:2.5,d:8.2,pin:[7.9,.5,1.5]}
};

/** Render the approved illustration without binding example rooms to real devices. */
export function createHouse({container,onReady=()=>{}}) {
  let renderer,scene,observer,materials={},rafId=0,pointerId=null,destroyed=false;
  const listeners=new AbortController();
  function destroy(){
    if(destroyed)return;
    destroyed=true;cancelAnimationFrame(rafId);observer?.disconnect();listeners.abort();
    const canvas=renderer?.domElement;
    if(pointerId!==null&&canvas?.hasPointerCapture?.(pointerId))canvas.releasePointerCapture(pointerId);
    const geometries=new Set(),allMaterials=new Set(Object.values(materials)),textures=new Set();
    scene?.traverse(object=>{
      if(object.geometry)geometries.add(object.geometry);
      for(const material of (Array.isArray(object.material)?object.material:[object.material]))if(material)allMaterials.add(material);
      object.shadow?.dispose();
    });
    for(const material of allMaterials)for(const value of Object.values(material))if(value?.isTexture)textures.add(value);
    for(const resource of [...textures,...geometries,...allMaterials])resource.dispose();
    renderer?.renderLists?.dispose();renderer?.dispose();renderer?.forceContextLoss();canvas?.remove();
    scene?.clear();scene=renderer=observer=null;materials={};
  }
  try {
    renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,preserveDrawingBuffer:true,powerPreference:'low-power'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,1.8));
  renderer.setClearColor(0x000000,0);
  renderer.shadowMap.enabled=true;
  renderer.shadowMap.type=THREE.PCFSoftShadowMap;
  renderer.outputColorSpace=THREE.SRGBColorSpace;
  renderer.toneMapping=THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure=1.12;
  container.append(renderer.domElement);
  scene=new THREE.Scene();
  const house=new THREE.Group();scene.add(house);
  const camera=new THREE.OrthographicCamera(-12,12,8,-8,.1,100);
  const target=new THREE.Vector3(.7,.1,0);
  let yaw=.68,tilt=.78,zoom=1.1,currentYaw=yaw,currentTilt=tilt,currentZoom=zoom;
  let width=0,height=0,dirty=true,dragging=false,startX=0,startY=0;
  const hemisphere=new THREE.HemisphereLight(0xc9ddff,0x243137,1.25);scene.add(hemisphere);
  const keyLight=new THREE.DirectionalLight(0xe0ecff,1.8);keyLight.position.set(-4,14,8);keyLight.castShadow=true;
  Object.assign(keyLight.shadow.camera,{left:-15,right:15,top:13,bottom:-13,near:1,far:38});
  keyLight.shadow.mapSize.set(2048,2048);keyLight.shadow.bias=-.0004;keyLight.shadow.normalBias=.03;
  scene.add(keyLight);
  const backLight=new THREE.DirectionalLight(0x688fff,1.2);backLight.position.set(1,6,-9);scene.add(backLight);
  materials={};
  const mat=(name,color,roughness=.7,metalness=.05)=>materials[name]??(materials[name]=new THREE.MeshStandardMaterial({color,roughness,metalness}));
  const wallMat=mat('wall',0x66788b,.88);const wallEdge=mat('edge',0x91a3b1,.65);
  const wood=mat('wood',0x887b65,.88);const darkWood=mat('darkWood',0x35414d,.85);
  const fabric=mat('fabric',0x72849b,.95);const cushion=mat('cushion',0x9dacb5,.95);
  const black=mat('black',0x121c2c,.55);const porcelain=mat('porcelain',0xb5c6cb,.3);
  const leafMat=mat('leaf',0x3c7364,.95);const trunkMat=mat('trunk',0x5d4d40,.9);
  function box(w,h,d,x,y,z,material,parent=house,cast=true){const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),material);m.position.set(x,y+h/2,z);m.castShadow=cast;m.receiveShadow=true;parent.add(m);return m;}
  function cyl(r1,r2,h,x,y,z,material,parent=house,segments=24){const m=new THREE.Mesh(new THREE.CylinderGeometry(r1,r2,h,segments),material);m.position.set(x,y+h/2,z);m.castShadow=true;m.receiveShadow=true;parent.add(m);return m;}
  function sphere(r,x,y,z,material,parent=house){const m=new THREE.Mesh(new THREE.SphereGeometry(r,16,12),material);m.position.set(x,y,z);m.castShadow=true;parent.add(m);return m;}
  function strip(a,b,color=0x61e5d1,r=.017,parent=house){const curve=new THREE.LineCurve3(new THREE.Vector3(...a),new THREE.Vector3(...b));const m=new THREE.Mesh(new THREE.TubeGeometry(curve,1,r,6,false),new THREE.MeshBasicMaterial({color}));parent.add(m);return m;}
  function wall(w,h,d,x,z){box(w,h,d,x,.04,z,wallMat);box(w+.015,.045,d+.015,x,h+.04,z,wallEdge);}
  function plant(x,z,scale=1,y=.05){cyl(.19*scale,.14*scale,.36*scale,x,y,z,mat('pot',0x87918b));cyl(.035*scale,.05*scale,.7*scale,x,y+.3*scale,z,trunkMat);for(let i=0;i<6;i++){const a=i*2.4;const l=sphere(.22*scale,x+Math.sin(a)*.14*scale,y+(.65+i*.07)*scale,z+Math.cos(a)*.14*scale,leafMat);l.scale.set(.6,1.6,.7);l.rotation.z=Math.sin(a)*.5;}}
  function chair(x,z,rotation=0){const g=new THREE.Group();g.position.set(x,0,z);g.rotation.y=rotation;house.add(g);box(.53,.13,.55,0,.45,0,fabric,g);box(.53,.48,.11,0,.48,-.25,fabric,g);for(const dx of [-.19,.19])for(const dz of [-.19,.19])cyl(.025,.02,.45,dx,0,dz,black,g);return g;}
  function table(w,d,x,z){box(w,.12,d,x,.7,z,wood);for(const dx of [-w/2+.13,w/2-.13])for(const dz of [-d/2+.13,d/2-.13])box(.07,.7,.07,x+dx,0,z+dz,black);}
  const base=box(16.1,.32,10.3,1,-.45,0,mat('base',0x182536,.65,.2));
  const baseEdges=new THREE.LineSegments(new THREE.EdgesGeometry(base.geometry),new THREE.LineBasicMaterial({color:0x4e817f,transparent:true,opacity:.5}));baseEdges.position.copy(base.position);house.add(baseEdges);
  box(15.9,.05,10.1,1,-.12,0,mat('ground',0x293d40,.95));
  box(12.4,.16,8.5,0,-.08,0,mat('foundation',0x4b6070));
  for(const [id,r] of Object.entries(ROOM_LAYOUT)){
    if(id==='garden')continue;
    const fm=new THREE.MeshStandardMaterial({color:id==='bathroom'?0x667d86:(id==='living'?0x697168:0x5f6868),roughness:.84,emissive:0x000000,emissiveIntensity:.04});
    box(r.w,.05,r.d,r.x,.03,r.z,fm);
    if(id!=='bathroom')for(let i=-Math.floor(r.d/.23)/2;i<r.d/.23/2;i++){
      box(r.w,.006,.012,r.x,.085,r.z+i*.23,mat('floorline',0x363e41,.95),house,false);
    }
  }
  box(2.9,.05,3.7,-.45,.03,-2.05,mat('diningfloor',0x65706d));
  wall(12.4,1.9,.14,0,-4.2);wall(.15,1.55,8.4,-6.2,0);wall(.15,1.55,8.4,6.2,0);
  wall(12.4,.25,.14,0,4.2);
  wall(.13,1.25,3.6,1.05,-2.35);wall(.13,.65,2.3,1.05,3.05);
  wall(4.9,.85,.13,3.65,.08);wall(.13,1.05,3.4,3.95,2.35);
  wall(3.6,1.2,.13,-4.3,.02);wall(1.3,.55,.13,.4,.02);
  const windowGlass=new THREE.MeshStandardMaterial({color:0x5d9ca5,transparent:true,opacity:.55,metalness:.3,roughness:.13,emissive:0x315a69,emissiveIntensity:.35});
  box(3.4,.9,.035,-3.8,.65,-4.28,windowGlass);box(3.5,1,.035,3.4,.6,-4.28,windowGlass);
  for(const x of [-5.5,-4.4,-3.3,-2.2,1.7,2.8,3.9,5.1])box(.035,.97,.06,x,.62,-4.3,black);
  const blinds=[];for(let i=0;i<8;i++){const slat=box(3.35,.045,.06,3.4,1.6-i*.11,-4.32,mat('blinds',0xb5bdba,.8));blinds.push(slat);}
  for(const z of [-2.5,-1.25,0,1.25,2.5])box(.04,.9,1.1,6.29,.5,z,windowGlass);
  strip([-6.23,.13,4.28],[6.25,.13,4.28],0x69e5dc,.018);
  strip([-6.23,.13,-4.28],[-6.23,.13,4.28],0x69e5dc,.018);
  strip([6.25,.13,-4.28],[6.25,.13,4.28],0x69e5dc,.018);
  // Living room: sectional, layered rug, coffee table, television and shelving.
  box(4.5,.025,2.85,-2.7,.1,2.25,mat('rug',0x364855,.98));
  for(let i=0;i<9;i++)box(4.2,.003,.012,-2.7,.128,1.05+i*.29,mat('rugthread',0x50616c,.97),house,false);
  box(3.5,.25,1.05,-3.1,.22,3.35,mat('sofaBase',0x435568,.96));
  box(3.5,.65,.22,-3.1,.25,3.84,fabric);box(.23,.56,1.08,-4.75,.25,3.38,fabric);
  box(.23,.56,1.08,-1.45,.25,3.38,fabric);
  for(let i=0;i<3;i++){box(.99,.2,.78,-4.16+i*1.05,.47,3.23,fabric);const pillow=box(.65,.35,.2,-4.1+i*1.05,.64,3.64,cushion);pillow.rotation.x=-.15;}
  box(.95,.24,1.5,-4.35,.22,2.34,fabric);box(.86,.17,1.4,-4.35,.46,2.3,fabric);
  cyl(.74,.74,.09,-2.45,.47,1.95,mat('tabletop',0x9a8c70));cyl(.42,.55,.32,-2.45,.15,1.95,black);
  box(.4,.04,.28,-2.55,.57,1.94,mat('book',0xb0bbbc));cyl(.065,.055,.12,-2.12,.57,2.1,porcelain);
  box(2.4,.45,.36,-2.05,.13,.3,darkWood);const tv=box(2.05,1.12,.055,-2.05,.7,.29,black);
  const screen=box(1.98,1.04,.015,-2.05,.74,.326,new THREE.MeshStandardMaterial({color:0x111d32,roughness:.18,metalness:.2,emissive:0x294269,emissiveIntensity:.4}));
  strip([-3.04,.78,.339],[-1.08,.78,.339],0x647cba,.007);
  plant(-5.55,3.4,1.2);plant(.5,3.5,.85);
  cyl(.37,.4,.6,-.05,.1,2.95,darkWood);cyl(.45,.45,.05,-.05,.7,2.95,wood);plant(-.05,2.95,.5,.76);
  // Kitchen: L-shaped cabinetry, stove, sink, island and bar stools.
  box(3.2,.8,.67,-4.25,.1,-3.68,mat('cabinet',0x7f9297));box(.7,.8,2.7,-5.68,.1,-2.3,mat('cabinet',0x7f9297));
  box(3.3,.09,.78,-4.23,.9,-3.65,porcelain);box(.8,.09,2.75,-5.65,.9,-2.3,porcelain);
  for(let i=0;i<5;i++){box(.02,.7,.025,-5.55+i*.65,.15,-3.32,black);box(.18,.025,.04,-5.25+i*.65,.7,-3.3,black);}
  box(.83,.025,.42,-4.8,1,-3.65,mat('sink',0x435b62,.3,.5));
  cyl(.018,.018,.34,-4.8,1,-3.83,mat('chrome',0xc0d0d1,.25,.8));strip([-4.8,1.34,-3.83],[-4.8,1.34,-3.59],0xbdcbd0,.02);
  box(.9,.018,.5,-3.1,1,-3.62,black);for(const x of [-3.33,-2.9])for(const z of [-3.77,-3.48])cyl(.105,.105,.012,x,1.02,z,mat('burner',0x50687c,.3,.5));
  box(.75,1.45,.73,-5.6,.1,-1.15,mat('fridge',0x687b8b,.3,.6));box(.025,1.3,.04,-5.43,.2,-.77,black);
  box(1.7,.75,.72,-3.55,.12,-1.3,darkWood);box(1.86,.1,.92,-3.55,.87,-1.3,porcelain);
  cyl(.2,.22,.05,-3.45,.99,-1.3,mat('fruitbowl',0xa78865));sphere(.085,-3.5,1.08,-1.3,mat('fruit',0xc49450));sphere(.07,-3.37,1.06,-1.29,leafMat);
  for(const x of [-4.05,-3.2]){cyl(.22,.22,.07,x,.55,-.47,wood);cyl(.035,.05,.55,x,.04,-.47,black);}
  strip([-5.7,1.01,-3.26],[-2.7,1.01,-3.26],0xffd5a0,.018);
  // Dining room.
  table(1.55,1.85,-.5,-2.4);chair(-1.55,-2.45,-Math.PI/2);chair(.55,-2.45,Math.PI/2);chair(-.5,-3.6,0);chair(-.5,-1.2,Math.PI);
  cyl(.12,.1,.25,-.5,.83,-2.4,porcelain);sphere(.2,-.5,1.1,-2.4,leafMat);
  // Bedroom.
  box(3.3,.022,2.8,3.45,.1,-2.3,mat('bedRug',0x434557,.98));
  box(2.6,.3,2.3,3.5,.17,-2.5,darkWood);box(2.55,.75,.16,3.5,.3,-3.7,mat('headboard',0x666576,.95));
  box(2.52,.24,2.2,3.5,.46,-2.48,mat('linen',0xc0c3ba,.98));box(2.55,.08,1.4,3.5,.69,-2.03,mat('blanket',0x8b7e9f,.99));
  for(const x of [2.83,4.13]){box(1,.16,.52,x,.7,-3.2,cushion);box(.57,.46,.58,x+(x<3?-1:1)*.8,.15,-3.15,wood);cyl(.09,.14,.25,x+(x<3?-1:1)*.8,.61,-3.15,black);cyl(.25,.17,.24,x+(x<3?-1:1)*.8,.86,-3.15,mat('lampshade',0xc7bb9f));}
  box(1.7,.75,.48,4.5,.1,-.38,wood);plant(1.65,-3.4,.85);
  // Study.
  box(2.25,.025,2.85,2.5,.1,2.3,mat('officeRug',0x394a53,.95));table(1.8,.65,2.5,3.55);chair(2.5,2.5,Math.PI);
  box(.85,.51,.045,2.6,.95,3.63,black);box(.77,.44,.015,2.6,.99,3.599,new THREE.MeshBasicMaterial({color:0x647aa0}));box(.08,.2,.12,2.6,.8,3.62,black);box(.55,.025,.2,2.5,.84,3.3,black);
  box(.22,.28,.3,1.82,.83,3.6,porcelain);box(.65,1.25,.32,1.45,.1,1.2,darkWood);
  for(let i=0;i<3;i++){box(.65,.05,.34,1.45,.35+i*.4,1.2,wood);for(let b=0;b<4;b++)box(.08,.22,.21,1.22+b*.13,.41+i*.4,1.2,mat('book'+b,[0x718e87,0x9f8e80,0x687b99,0xb9b3a3][b]));}
  plant(3.5,3.7,.7);
  // Bathroom: walk-in glass shower, sink and tub.
  for(let i=0;i<5;i++)for(let j=0;j<8;j++)box(.385,.007,.435,4.25+i*.39,.093,.48+j*.44,mat('tile',((i+j)%2)?0x6c8087:0x73888b),house,false);
  box(1.45,.42,.72,5.1,.12,3.42,porcelain);box(1.22,.03,.48,5.1,.55,3.42,mat('water',0x75adb3,.05,.35));
  box(1.3,.65,.42,5.3,.1,.52,wood);box(1.38,.07,.48,5.3,.75,.52,porcelain);cyl(.22,.17,.09,5.3,.82,.52,porcelain);
  box(.035,1.15,.78,4.1,.1,1.15,windowGlass);cyl(.15,.15,.4,5.32,.15,1.75,porcelain);sphere(.26,5.32,.57,1.72,porcelain).scale.set(1,.3,1.3);
  // Terrace, pool and garden.
  box(2.5,.04,8.2,7.67,-.035,0,mat('terrace',0x61716a,.95));
  for(let i=0;i<25;i++)box(2.45,.009,.018,7.67,.008,-3.95+i*.33,mat('deckline',0x2d4c4a),house,false);
  box(1.95,.12,3.55,7.7,.03,-1.82,mat('poolRim',0x819b99));box(1.71,.04,3.3,7.7,.16,-1.82,new THREE.MeshStandardMaterial({color:0x28787d,metalness:.3,roughness:.12,emissive:0x1d8b95,emissiveIntensity:.35}));
  for(let i=0;i<8;i++)strip([6.89,.203,-3.35+i*.4],[8.51,.203,-3.35+i*.4],0x5ca6aa,.006);
  strip([6.85,.21,-3.49],[6.85,.21,-.15],0x7ff5df,.018);strip([8.55,.21,-3.49],[8.55,.21,-.15],0x7ff5df,.018);
  box(.64,.18,1.55,7.17,.08,2.23,wood);box(.59,.12,1.48,7.17,.26,2.23,cushion);box(.65,.18,1.55,8.12,.08,2.23,wood);box(.6,.12,1.48,8.12,.26,2.23,cushion);
  cyl(.32,.32,.06,7.65,.49,1.1,wood);cyl(.04,.07,.45,7.65,.04,1.1,black);
  for(const [x,z,s] of [[-6.65,-4.6,1.3],[8.7,4.1,1],[8.65,-4.35,1.3],[-6.65,4.6,.75]])plant(x,z,s,-.07);
  for(let i=0;i<9;i++){const shrub=sphere(.28,-4.8+i*1.25,.1,-4.7,leafMat);shrub.scale.set(1.4,.6,.8);}
  const airUnitPositions={living:[-6.05,1.15,2.0,Math.PI/2],kitchen:[-4.2,1.48,-4.04,0],bedroom:[3.5,1.48,-4.04,0],office:[3.81,.66,2.0,-Math.PI/2],bathroom:[6.04,1.05,2.0,-Math.PI/2]};
  function createAirUnit(id){
    if(!airUnitPositions[id])return null;
    const [x,y,z,rotation]=airUnitPositions[id];const group=new THREE.Group();group.position.set(x,y,z);group.rotation.y=rotation;house.add(group);
    box(1.05,.24,.2,0,0,0,mat('acBody',0xaebec6,.45),group);box(.83,.025,.025,0,.045,.108,black,group);
    for(let i=0;i<4;i++)box(.78,.009,.025,0,.087+i*.025,.108,mat('acGrille',0x637b8b,.6),group,false);
    const led=box(.055,.025,.015,.4,.19,.112,new THREE.MeshBasicMaterial({color:0x71bdff}),group,false);
    const air=new THREE.Group();group.add(air);const lines=[];
    for(let i=0;i<3;i++){
      const dx=(i-1)*.24;const curve=new THREE.QuadraticBezierCurve3(new THREE.Vector3(dx,.035,.15),new THREE.Vector3(dx,-.14,.6),new THREE.Vector3(dx,-.48,.84));
      const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(16)),new THREE.LineBasicMaterial({color:0x71bdff,transparent:true,opacity:.75}));air.add(line);lines.push(line);
    }
    return {group,led,air,lines};
  }
  for (const id of Object.keys(airUnitPositions)) { const unit=createAirUnit(id); unit.led.material.color.set(0x42576d); unit.air.visible=false; }
  function resize(){if(destroyed)return;width=container.clientWidth;height=container.clientHeight;if(!width||!height)return;renderer.setSize(width,height,false);dirty=true;}
  observer=new ResizeObserver(resize);observer.observe(container);resize();
  const boundsCorners=[];
  for(const x of [-7.25,9.25])for(const y of [-.5,2.2])for(const z of [-5.3,5.3])boundsCorners.push(new THREE.Vector3(x,y,z));
  const cameraPoint=new THREE.Vector3();
  const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)').matches;
  function cameraUpdate(){
    const a=width/height;
    camera.position.set(target.x+26*Math.sin(currentTilt)*Math.sin(currentYaw),26*Math.cos(currentTilt),target.z+26*Math.sin(currentTilt)*Math.cos(currentYaw));camera.lookAt(target);camera.updateMatrixWorld();
    let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
    for(const corner of boundsCorners){cameraPoint.copy(corner).applyMatrix4(camera.matrixWorldInverse);minX=Math.min(minX,cameraPoint.x);maxX=Math.max(maxX,cameraPoint.x);minY=Math.min(minY,cameraPoint.y);maxY=Math.max(maxY,cameraPoint.y);}
    const span=Math.max((maxY-minY)*1.16,(maxX-minX)*1.14/a)/currentZoom;
    const cx=(minX+maxX)/2,cy=(minY+maxY)/2;
    camera.left=cx-span*a/2;camera.right=cx+span*a/2;camera.top=cy+span/2;camera.bottom=cy-span/2;camera.updateProjectionMatrix();
  }
  let lastTime=0;
  function frame(time){
    if(destroyed)return;
    rafId=requestAnimationFrame(frame);
    if(document.hidden||time-lastTime<30)return;
    lastTime=time;
    const diff=Math.abs(currentYaw-yaw)+Math.abs(currentTilt-tilt)+Math.abs(currentZoom-zoom);
    if(diff>.0001){const ease=reducedMotion?1:.13;currentYaw+=(yaw-currentYaw)*ease;currentTilt+=(tilt-currentTilt)*ease;currentZoom+=(zoom-currentZoom)*ease;dirty=true;}
    if(!dirty)return;cameraUpdate();renderer.render(scene,camera);dirty=false;
  }
  rafId=requestAnimationFrame(frame);
  const canvas=renderer.domElement;
  const listen=(event,callback,options={})=>canvas.addEventListener(event,callback,{...options,signal:listeners.signal});
  listen('pointerdown',event=>{if(event.button!==0)return;pointerId=event.pointerId;dragging=true;startX=event.clientX;startY=event.clientY;canvas.setPointerCapture(pointerId);});
  listen('pointermove',event=>{if(!dragging)return;const dx=event.clientX-startX,dy=event.clientY-startY;yaw-=dx*.006;tilt=THREE.MathUtils.clamp(tilt+dy*.003,.12,1.18);startX=event.clientX;startY=event.clientY;dirty=true;});
  const stopDrag=()=>{dragging=false;if(pointerId!==null&&canvas.hasPointerCapture(pointerId))canvas.releasePointerCapture(pointerId);pointerId=null;};
  listen('pointerup',stopDrag);listen('pointercancel',stopDrag);listen('lostpointercapture',()=>{dragging=false;pointerId=null;});
  listen('wheel',event=>{if(event.ctrlKey||event.metaKey){event.preventDefault();zoom=THREE.MathUtils.clamp(zoom-event.deltaY*.002,.65,1.65);dirty=true;}},{passive:false});
  listen('webglcontextlost',event=>{event.preventDefault();onReady(false);});
  listen('webglcontextrestored',()=>{dirty=true;onReady(true);});
  onReady(true);
  return {
    setView(view){if(destroyed)return;tilt=view==='top'?.025:.78;yaw=.68;zoom=1.1;dirty=true;},
    zoom(delta){if(destroyed)return;zoom=THREE.MathUtils.clamp(zoom+delta,.65,1.65);dirty=true;},
    reset(){if(destroyed)return;yaw=.68;tilt=.78;zoom=1.1;dirty=true;},destroy
  };
  } catch {
    destroy();
    const fallback=document.createElement('div');fallback.className='house-fallback';
    fallback.innerHTML='<svg viewBox="0 0 800 480" aria-hidden="true"><g transform="translate(390 220) scale(1 .58) rotate(30)"><rect x="-260" y="-180" width="520" height="360" rx="5" fill="#233545" stroke="#62e8cd" stroke-width="2"/><path d="M-260-180h520v360M-260-180v360M-260 0h520M20-180v360M150 0v180M-100-180V0" fill="none" stroke="#9caabc" stroke-width="14"/><rect x="-220" y="50" width="160" height="60" rx="9" fill="#7c8fb0"/><rect x="60" y="-130" width="140" height="95" rx="7" fill="#c4b89a"/></g></svg>';
    container.append(fallback);onReady(false);
    return {setView(){},zoom(){},reset(){},destroy(){fallback.remove();}};
  }
}
