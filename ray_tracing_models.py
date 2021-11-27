import taichi as ti
from typing import List

PI = 3.14159265

@ti.func
def rand3():
    return ti.Vector([ti.random(), ti.random(), ti.random()])

@ti.func
def random_in_unit_sphere():
    p = 2.0 * rand3() - ti.Vector([1, 1, 1])
    while p.norm() >= 1.0:
        p = 2.0 * rand3() - ti.Vector([1, 1, 1])
    return p

@ti.func
def random_in_unit_disk():
    p = 2.0 * ti.Vector([ti.random(), ti.random(), 0]) - ti.Vector([1, 1, 0])
    while p.norm() >= 1.0:
        p = 2.0 * ti.Vector([ti.random(), ti.random(), 0]) - ti.Vector([1, 1, 0])
    return p    

@ti.func
def random_unit_vector():
    return random_in_unit_sphere().normalized()

@ti.func
def to_light_source(hit_point, light_source):
    return light_source - hit_point

@ti.func
def reflect(v, normal):
    return v - 2 * v.dot(normal) * normal

@ti.func
def refract(uv, n, etai_over_etat):
    cos_theta = min(n.dot(-uv), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.dot(r_out_perp))) * n
    return r_out_perp + r_out_parallel

@ti.func
def reflectance(cosine, ref_idx):
    # Use Schlick's approximation for reflectance.
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * pow((1 - cosine), 5)

@ti.data_oriented
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
    def at(self, t):
        return self.origin + t * self.direction

@ti.data_oriented
class HitRecord:
    def __init__(self, is_hit=False, hit_time=0.0, hit_point=None, hit_point_normal=None, 
                 is_front_face=None, material=None, color=None):
        self.is_hit = is_hit
        self.hit_time = hit_time
        self.hit_point = hit_point
        self.hit_point_normal = hit_point_normal
        self.is_front_face = is_front_face
        self.material = material
        self.color = color
        
@ti.data_oriented
class Hittable_list:
    def __init__(self):
        self.objects = []
    def add(self, obj):
        self.objects.append(obj)
    def clear(self):
        self.objects = []

    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):
        closest_t = t_max
        is_hit = False
        front_face = False
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        color = ti.Vector([0.0, 0.0, 0.0])
        material = 1
        for index in ti.static(range(len(self.objects))):
            # is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp =  self.objects[index].hit(ray, t_min, closest_t)
            hitrec =  self.objects[index].hit(ray, t_min, closest_t)
            if hitrec.is_hit:
                closest_t = hitrec.hit_time
                is_hit = hitrec.is_hit
                hit_point = hitrec.hit_point
                hit_point_normal = hitrec.hit_point_normal
                front_face = hitrec.is_front_face
                material = hitrec.material
                color = hitrec.color
        return HitRecord(is_hit, 0, hit_point, hit_point_normal, front_face, material, color)

    @ti.func
    def hit_shadow(self, ray, t_min=0.001, t_max=10e8):
        is_hit_source = False
        is_hit_source_temp = False
        hitted_dielectric_num = 0
        is_hitted_non_dielectric = False
        # Compute the t_max to light source
        # is_hit_tmp, root_light_source, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp = \
        hitrec = self.objects[0].hit(ray, t_min)
        for index in ti.static(range(len(self.objects))):
            # is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp =  self.objects[index].hit(ray, t_min, root_light_source)
            hitrec =  self.objects[index].hit(ray, t_min, hitrec.hit_time)
            is_hit_tmp = hitrec.is_hit
            material_tmp = hitrec.material
            if is_hit_tmp:
                if material_tmp != 3 and material_tmp != 0:
                    is_hitted_non_dielectric = True
                if material_tmp == 3:
                    hitted_dielectric_num += 1
                if material_tmp == 0:
                    is_hit_source_temp = True
        if is_hit_source_temp and (not is_hitted_non_dielectric) and hitted_dielectric_num == 0:
            is_hit_source = True
        return is_hit_source, hitted_dielectric_num, is_hitted_non_dielectric


@ti.data_oriented
class Camera:
    def __init__(self, fov=60, aspect_ratio=1.0, aperture=0.0):
        # Camera parameters
        self.lookfrom = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.lookat = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.vup = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.fov = fov
        self.aspect_ratio = aspect_ratio

        self.cam_lower_left_corner = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_horizontal = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_vertical = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_origin = ti.Vector.field(3, dtype=ti.f32, shape=())
        
        # init setting
        self.lookfrom[None] = [0.0, 1.0, -5.0]
        self.reset(0.0, 1.0, -1.0, 0)
        
        # defocus blur
        self.aperture = aperture
        self.lens_radius = self.aperture / 2
        
    @ti.kernel
    def reset(self, x: ti.f32, y: ti.f32, z: ti.f32, t: ti.u8):
        if t == 0:  # change lookat
            self.lookat[None] = [x, y, z]
        elif t == 1: # change lookfrom
            self.lookfrom[None] = [x, y, z]
        self.vup[None] = [0.0, 1.0, 0.0]
        theta = self.fov * (PI / 180.0)
        half_height = ti.tan(theta / 2.0)
        half_width = self.aspect_ratio * half_height
        self.cam_origin[None] = self.lookfrom[None]
        
        w = (self.lookfrom[None] - self.lookat[None]).normalized()
        u = (self.vup[None].cross(w)).normalized()
        v = w.cross(u)
        
        self.cam_lower_left_corner[None] = ti.Vector([-half_width, -half_height, -1.0])
        self.cam_lower_left_corner[
            None] = self.cam_origin[None] - half_width * u - half_height * v - w
        self.cam_horizontal[None] = 2 * half_width * u
        self.cam_vertical[None] = 2 * half_height * v

    @ti.func
    def get_ray(self, u, v):
        w = (self.lookfrom[None] - self.lookat[None]).normalized()
        s = (self.vup[None].cross(w)).normalized()
        t = w.cross(s)
        
        rd = self.lens_radius * random_in_unit_disk()
        offset = s * rd.x + t * rd.y
        return Ray(self.cam_origin[None] + offset, 
                   self.cam_lower_left_corner[None] + u * self.cam_horizontal[None] + v * self.cam_vertical[None] - self.cam_origin[None] - offset)
    