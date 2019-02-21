#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;

in vec2 fs_Pos;
out vec4 out_Col;


#define NUM_NOISE_OCTAVES 7
#define NUM_T_MARCH_STEPS 80
#define FOV 45.f
#define PI 3.1415926f
#define DRAG_MULT 0.018
#define ITERATIONS_RAYMARCH 6
#define ITERATIONS_NORMAL 26
#define WATER_DEPTH 1

vec3 lcol = vec3(1.0,0.5,0.0);

vec3 getld(){
    return  (vec3(0.f,1.2f,15.f*cos(u_Time/200.f) +  15.f));
}
vec2 rotate(vec2 p, float a)
{
    float sa = sin(a);
    float ca = cos(a);
    return vec2(ca*p.x + sa*p.y, -sa*p.x + ca*p.y);
}

float hash(float n) { return fract(sin(n) * 1e4); }
float hash(vec2 p) { return fract(1e4 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x)))); }
float random (in vec2 st) {
    return fract(sin(dot(st.xy,
                         vec2(12.9898,78.233)))*
        43758.5453123);
}

float vmax(vec2 v) {
	return max(v.x, v.y);
}

float vmax(vec3 v) {
	return max(max(v.x, v.y), v.z);
}

float vmax(vec4 v) {
	return max(max(v.x, v.y), max(v.z, v.w));
}


float noise (in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

float noise(vec3 x) {
    const vec3 step = vec3(110, 241, 171);

    vec3 i = floor(x);
    vec3 f = fract(x);

    // For performance, compute the base input to a 1D hash from the integer part of the argument and the
    // incremental change to the 1D based on the 3D -> 1D wrapping
    float n = dot(i, step);

    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix( hash(n + dot(step, vec3(0, 0, 0))), hash(n + dot(step, vec3(1, 0, 0))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 0))), hash(n + dot(step, vec3(1, 1, 0))), u.x), u.y),
               mix(mix( hash(n + dot(step, vec3(0, 0, 1))), hash(n + dot(step, vec3(1, 0, 1))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 1))), hash(n + dot(step, vec3(1, 1, 1))), u.x), u.y), u.z);
}

#define OCTAVES 3
float fbm (in vec2 st) {
    // Initial values
    float value = 0.0;
    float amplitude = .5;
    float frequency = 0.;
    //
    // Loop of octaves
    for (int i = 0; i < OCTAVES; i++) {
        value += amplitude * noise(st);
        st *= 2.;
        amplitude *= .5;
    }
    return value;
}

float fbm(vec3 x) {
	float v = 0.0;
	float a = 0.5;
	vec3 shift = vec3(100);
	for (int i = 0; i < OCTAVES; ++i) {
		v += a * noise(x);
		x = x * 2.0 + shift;
		a *= 0.5;
	}
	return v;
}


///==============================================
float Plane( vec3 p, vec4 n )
{
  // n must be normalized
  return dot(p,n.xyz) + n.w;
}
float Cylinder(vec3 p, float r, float height) {
	float d = length(p.xz) - r;
	d = max(d, abs(p.y) - height);
	return d;
}
float Sphere( vec3 p, float s )
{
  float c = Cylinder(p+vec3(0.f,-2.0f,0.f),3.f,2.f);

  p.x = abs(p.x) + .9;
  float ss = length(p)-s;
  return max(-c,ss);
}

float Box( vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return length(max(d,0.0))
         + min(max(d.x,max(d.y,d.z)),0.0);
}



float rotbox(vec3 p){
    vec3 rotp = p;
    rotp.xz = rotate(rotp.xz,PI/4.f);
    float b1 = Box(p,vec3(1.2,15,1.2));
    float b2 = Box(rotp,vec3(1.2,15,1.2));
    return  max(b2,b1);
}

float HexPrism( vec3 p, vec2 h )
{
    const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
    p = abs(p);
    p.xy -= 2.0*min(dot(k.xy, p.xy), 0.0)*k.xy;
    vec2 d = vec2(
       length(p.xy-vec2(clamp(p.x,-k.z*h.x,k.z*h.x), h.x))*sign(p.y-h.x),
       p.z-h.y );
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float TriPrism( vec3 p, vec2 h )
{
    vec3 q = abs(p);
    return max(q.z-h.y,max(q.x*0.866025+p.y*0.5,-p.y)-h.x*0.5);
}

float Octahedron( in vec3 p, in float s)
{
    p = abs(p);
    float m = p.x+p.y+p.z-s;
    vec3 q;
         if( 3.0*p.x < m ) q = p.xyz;
    else if( 3.0*p.y < m ) q = p.yzx;
    else if( 3.0*p.z < m ) q = p.zxy;
    else return m*0.57735027;

    float k = clamp(0.5*(q.z-q.y+s),0.0,s);
    return length(vec3(q.x,q.y-s+k,q.z-k));
}

float hshape(vec3 p){
        vec3 hh = vec3(-.52f,-.8f,0.f);
        vec2 hhi = vec2(1.f,0.5f)*p.xy;
        vec2 hprot = rotate(hhi,PI/6.f);
        vec3 hp = vec3(hprot.x,hprot.y,p.z);
        hp += hh;
        float f = 1e10;
        float h0 = HexPrism(hp,vec2(1.4f,1.8f));
        float h1 = HexPrism(hp,vec2(1.6f,1.6f));
        float h2 = HexPrism(hp,vec2(1.8f,1.4f));
        f = min(f,h1);
        f = min(f,h2);
        f = min(f,h0);
        return f;
}

float stage2s(vec3 p){

        float f = 1e10;
        float h = hshape(p);
        vec2 h2rot = rotate(p.xz,PI/2.f);
        vec3 h2roted = vec3(h2rot.x,p.y,h2rot.y);
        float h2 = hshape(h2roted);
        f = min(f,h);
        f = min(f,h2);
        return f;
}

float stage1s(vec3 p){

    float f = 1e10;
    float b1 = Box(p,vec3(2));
    f = min(b1,f);
    float b2 = Box(p,vec3(2.2,1.8,1.8));
    float b3 = Box(p,vec3(1.8,1.8,2.2));
    float b4 = Box(p,vec3(2.4,1.6,1.6));
    float b5 = Box(p,vec3(1.6,1.6,2.4));
    f = min(f,b2);
    f = min(f,b3);
    f = min(f,b4);
    f = min(f,b5);
    return f;

}

float stage3s(vec3 p){

    float f = 1e10;
    vec3 stage3dis = vec3(0.f,-1.9f,0.f);
    vec3 stage3p = p+stage3dis;
    stage3p.y*=2.8f;
    stage3p.x*=1.06f;
    stage3p.z/=1.2f;
    float t1 = TriPrism(stage3p,vec2(2.f));
    stage3p+=vec3(0.f,-0.5f,0.f);
    stage3p.x/=1.1f;
    stage3p.z*=1.08f;
    float t2 = TriPrism(stage3p,vec2(2.f));
    stage3p+=vec3(0.f,-0.5,0.f);
    stage3p.x/=1.1f;
    stage3p.z*=1.08f;
    float t3 = TriPrism(stage3p,vec2(2.f));
    f = min(f,t3);
    f = min(f,t2);
    f = min(f,t1);
    return f;
}

bool aabbBoxIntersect(vec3 ro,vec3 rd, vec3 minb, vec3 maxb)
{
	float tnear = -1e10;
	float tfar = 1e10;

    float EPSILON = 1e-4;

	for (int i = 0; i<3; i++)
	{
		float t0, t1;

		if (abs(rd[i]) < EPSILON)
		{
			if (ro[i] < minb[i] || ro[i] > maxb[i])
				return false;
			else
			{
				t0 = -1e10;
				t1 = 1e10;
			}
		}
		else
		{
			t0 = (minb[i] - ro[i]) / rd[i];
			t1 = (maxb[i] - ro[i]) / rd[i];
		}

		tnear = max(tnear, min(t0, t1));
		tfar = min(tfar, max(t0, t1));
	}

	if (tfar < tnear) return false; // no intersection

	if (tfar < 0.f) return false; // behind origin of ray

	return true;

}

float compshape(vec3 p){

    float f = 1e10;
    //float c1 = Cylinder(p,1.2f,40.f);
    float c1 = rotbox(p);
    f = min(c1,f);
    float stage1 = stage1s(p);
    f = min(stage1,f);

    vec3 stage2dis = vec3(0.f,-2.f,0.f);

    vec3 stage2p = stage2dis+p;
    stage2p.xz*=1.3f;
    stage2p.y/=1.3f;
    float stage2 = stage1s(stage2p);
    f = min(f,stage2);

    float stage3 = stage3s(stage2p);
    f = min(f,stage3);

    stage2p.xz = rotate(stage2p.xz,PI/2.f);
    float stage3rot = stage3s(stage2p);
    f = min(f,stage3rot);

    vec3 stage4dis = vec3(0.f,-6.f,0.f);
    vec3 stage4p = stage4dis+p;
    stage4p.xz*=1.8f;
    float stage4 = stage1s(stage4p);
    f = min(f,stage4);

    return f;
}

float repeatxz(in vec3 p, in vec2 xz){

    p.y = -abs(-p.y+11.f) + 11.f;
    vec2 modxz = mod(p.xz,xz);
    vec3 q = vec3(modxz.x,p.y,modxz.y) - 0.5f*vec3(xz.x,0.f,xz.y);
    return compshape(q);
}


float scene(vec3 pos){

    float d = 1e10;
    float cylinder1 = repeatxz(pos,vec2(13.f,13.f));
    float boat = Sphere(pos-getld()+vec3(0.f,.5f,0.f),1.2f);
    float cy2 = Cylinder(pos-getld()+vec3(0.f,0.3f,0.f),0.02f,.25f);
    float plane = Plane(-pos,vec4(0.f,1.f,0.f,23.f));
    d = min(d,cylinder1);
    d = min(d,boat);
    d = min(d,plane);
    d = min(d,cy2);
    //float plane = Plane(pos,vec4(0.0,1.0,0.0,2.f));
    //d = min(d,plane);
    return d;
}


float softshadow( in vec3 ro, in vec3 rd, float mint, float maxt, float k ,vec3 lp)
{
   float res = 1.0;
    float ph = 1e20;
    float t = mint;
    for( t=mint; t < maxt; )
    {
        float h = scene(ro + rd*t);
        if( h<0.0001){
            if(length(ro-lp)>length(rd*t))
            return 0.0;
            else
            return 1.0;
            }

        float y = h*h/(2.0*ph);
        float d = sqrt(h*h-y*y);
        res = min( res, k*d/max(0.0,t-y) );
        ph = h;
        t += h;
    }
    if(length(ro-lp)<length(rd*t)) return 1.f;
    return res;
}


vec3 calcNormal( in vec3 pos, float col )
{
    const float eps = 0.001;

    const vec3 v1 = vec3( 1.0,-1.0,-1.0);
    const vec3 v2 = vec3(-1.0,-1.0, 1.0);
    const vec3 v3 = vec3(-1.0, 1.0,-1.0);
    const vec3 v4 = vec3( 1.0, 1.0, 1.0);

	vec3 res = normalize( v1*scene( pos + v1*eps ) +
					  v2*scene( pos + v2*eps ) +
					  v3*scene( pos + v3*eps ) +
					  v4*scene( pos + v4*eps ) );


	return res ;
}


float calcIntersection( in vec3 ro, in vec3 rd )
{


	const float maxd = 180.0;
	const float precis = 0.005;
    float h = precis*2.0;
    float t = 0.0;
	float res = -1.0;
    for( int i=0; i<60; i++ )
    {
        if( abs(h)<precis||t>maxd ) break;
	    h = scene(ro+rd*t);
        t += h;
    }

    if( t<maxd ) res = t;
    return res;
}

float calcAO( in vec3 pos, in vec3 nor )
{
	float occ = 0.0;
    float sca = 2.0;
    for( int i=0; i<5; i++ )
    {
        float h = 0.001 + 0.15*float(i)/4.0;
        float d = scene( pos + h*nor );
        occ += (h-d)*sca;
        sca *= 0.95;
    }
    return clamp( 1.0 - 1.5*occ, 0.0, 1.0 );
}

vec3 render(in vec3 ro, in vec3 rd, inout float t){

       vec3 rot = ro;
       //vec2 modxz = mod(rot.xz,vec2(15.f,15.f));
       //vec3 q = vec3(modxz.x,rot.y,modxz.y) - 0.5f*vec3(15.f,0.f,15.f);
       //vec3 rdt = normalize()
      //bool intersect = aabbBoxIntersect(ro,rd,)
      vec3 col = vec3(0.f);
      float res = calcIntersection(ro,rd);
      t = res;
      if(res>0.f){
            vec3 curp = res*rd + ro;
            float    texcol = mix(0.f,fbm(curp),(curp.y)/50.f);
            vec3 nor = calcNormal(curp,texcol);
            vec3 ld = normalize(getld()-curp);
            float ldis = length(getld()-curp);
            float ldisfac = 10.f/pow(1.f+ldis,.91);
            float ssd = softshadow(curp,ld,0.01f,70.f,20.f,getld());
            vec3 ld2 = normalize(vec3(1,-1,1));
            float lamb2 = dot(ld2,nor);
            float l = dot(ld,nor);
            float ao = calcAO(curp,nor);
            col = ao*ssd*l*lcol*ldisfac+vec3(texcol-.5f)/6.f;
            col += (vec3(0.f,0.05f,0.05)+vec3(0.f,0.07f,0.2f)*lamb2)*ao;//ambient ligh t?
      }
      return col;
}




vec2 wavedx(vec2 position, vec2 direction, float speed, float frequency, float timeshift) {
    float x = dot(direction, position) * frequency + timeshift * speed;
    float wave = exp(sin(x) - 1.0);
    float dx = wave * cos(x);
    return vec2(wave, -dx);
}

float getwaves(vec2 position, int iterations){
	float iter = 0.0;
    float phase = 6.0;
    float speed = 2.0;
    float weight = 1.0;
    float w = 0.0;
    float ws = 0.0;
    for(int i=0;i<iterations;i++){
        vec2 p = vec2(sin(iter), cos(iter));
        vec2 res = wavedx(position, p, speed, phase, u_Time/100.f);
        position += normalize(p) * res.y * weight * DRAG_MULT;
        w += res.x * weight;
        iter += 12.0;
        ws += weight;
        weight = mix(weight, 0.0, 0.2);
        phase *= 1.18;
        speed *= 1.07;
    }
    return w / (ws*4.f);
}

vec3 normalw(vec2 pos, float e, float depth){
    vec2 ex = vec2(e, 0);
    float eps = 0.1;
    float H = getwaves(pos.xy * eps, ITERATIONS_NORMAL) * depth;
    vec3 a = vec3(pos.x, H, pos.y);
    return normalize(cross(normalize(a-vec3(pos.x - e, getwaves(pos.xy * eps - ex.xy * eps, ITERATIONS_NORMAL) * depth, pos.y)),
                           normalize(a-vec3(pos.x, getwaves(pos.xy * eps + ex.yx * eps, ITERATIONS_NORMAL) * depth, pos.y + e))));
}


//worly noise

float fbmw(vec2 x) {
	float v = 0.0;
	float a = .3;
	vec2 shift = vec2(1)*u_Time/150.f;
	for (int i = 0; i < 3; ++i) {

	    x = rotate(x,45.f) * 1.4 + 5.f*shift/pow(float(i)+1.f,0.3);
		v += a * noise(x);
        x *= 1.2;
		a *= 0.5;
	}
	return pow(v,4.f);
}

float heightfield(vec2 p){
    return getwaves(p,10);
}

float fbmsdf(vec3 p){
    float curH = heightfield(p.xz);
    curH = p.y-curH;
    return curH;
}
float softshadowT(in vec3 ro, in vec3 rd, float mint, float maxt, float k){
        float res = 1.0;
        float ph = 1e20;
        for( float t=mint; t < maxt; )
        {
            float h = fbmsdf(ro + rd*t);
            if( h<0.001 )
                return 0.0;
            float y = h*h/(2.0*ph);
            float d = sqrt(h*h-y*y);
            res = min( res, k*d/max(0.0,t-y) );
            ph = h;
            t += h;
        }
        return res;
}

float raymarchT(vec3 ro, vec3 rd){
    float res = -1.f;
    float precis = 0.00002;
    float maxT = 80.f;
    float t = 0.f;
    float curH = 2.f*precis;
    for(int i = 0;i<NUM_T_MARCH_STEPS;i++){
        vec3 curpos = ro+rd*t;
        if(curH>=maxT||abs(curH)<=precis||t>80.f) {
            break;
        }
        curH = fbmsdf(curpos);
        t+=curH;
    }
    if(curH<maxT&&t<80.f) res = t;
    return res;
}



vec3 compnor(vec3 pos){
    float thres = 0.01;
    float nx = heightfield(vec2(pos.x+thres,pos.z));
    float ny = heightfield(vec2(pos.x,pos.z+thres));
    vec3 px = vec3(pos.x+thres,nx,pos.z);
    vec3 py = vec3(pos.x,ny,pos.z+thres);
    vec3 nor = -cross(px-pos,py-pos);
    return normalize(nor);
}

vec3 rendert(in vec3 ro, in vec3 rd, in float t, inout float selft){
    float tv = raymarchT(ro,rd);
    selft = tv;
    if(tv>t&&t!=-1.f) return vec3(0);
    vec3 col = vec3(0);
    if(tv>=0.f ){
        vec3 curp = ro + tv*rd;
        vec3 nor = normalw(curp.xz,0.01,1.0);
        vec3 rfl = normalize(reflect(rd,-nor));
        vec3 ldi = normalize(getld()-curp);
        float vv = dot(rfl,ldi);
        if(vv>0.99999){
                return lcol+vec3(0.f,0.7,0.3);

        }
        float st = -1.f;
        vec3 rlcol = render(curp,rfl,st);
        vec3 ld = normalize(getld()-curp);
        float lamb = dot(ld,nor);
        //float ssd = softshadow(curp,ld,0.001f,20.f,88.f,getld());
        vec3 acol = mix(vec3(0.f,0.1,0.1),vec3(0.f,0.2,0.3),curp.y);
        col= 0.8*rlcol+0.2*acol;//vec3(1) *ssd*lamb;
    }
    return col;
}

void main() {

  float sx = (2.f*gl_FragCoord.x/u_Dimensions.x)-1.f;
  float sy = 1.f-(2.f*gl_FragCoord.y/u_Dimensions.y);
  float len = length(u_Ref - u_Eye);
  vec3 forward = normalize(u_Ref - u_Eye);
  vec3 right = cross(forward,u_Up);
  vec3 V = u_Up * len * tan(FOV/2.f);
  vec3 H = right * len * (u_Dimensions.x/u_Dimensions.y) * tan(FOV/2.f);
  vec3 p = u_Ref + sx * H - sy * V;





  vec3 ro = u_Eye;
  vec3 rd = normalize(p - u_Eye);


   vec3 lp = getld();//light point
   vec3 ld = normalize(lp - ro);
   float vv = dot(rd,ld);
   if(vv>0.999999){
        out_Col = mix(vec4(0.f,0.f,0.f,1.f),vec4(lcol+vec3(0.f,0.7,0.3),1.f),(vv-0.9999)*10000.f);
        return;
   }


  vec3 colw = vec3(0.f,0.03f,0.1f);
  float st = -1.f;
  vec3 col = render(ro,rd,st);//render(ro,rd);
  float wt = -1.f;
  colw = rendert(ro,rd,st,wt);
  vec3 finalcol = col;
  if(wt<=st&&wt!=-1.f||st==-1.f)
  finalcol = colw;


  out_Col = vec4(finalcol,1);
}
