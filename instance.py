import taichi as ti
from ray_tracing_models import HitRecord, Hittable_list, Ray

PI = 3.14159265

@ti.data_oriented
class Sphere:
    def __init__(self, center, radius, material, color):
        self.center = center
        self.radius = radius
        self.material = material
        self.color = color

    @ti.func
    def hit(self, ray, t_min=1e-3, t_max=10e8):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        is_hit = False
        front_face = False
        root = 0.0
        hit_point =  ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        if discriminant > 0:
            sqrtd = ti.sqrt(discriminant)
            root = (-b - sqrtd) / (2 * a)
            if root < t_min or root > t_max:
                root = (-b + sqrtd) / (2 * a)
                if root >= t_min and root <= t_max:
                    is_hit = True
            else:
                is_hit = True
        if is_hit:
            hit_point = ray.at(root)
            hit_point_normal = (hit_point - self.center) / self.radius
            # Check which side does the ray hit, we set the hit point normals always point outward from the surface
            if ray.direction.dot(hit_point_normal) < 0:
                front_face = True
            else:
                hit_point_normal = -hit_point_normal
        return HitRecord(is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color)
    
@ti.data_oriented
class Box_old:
    # construct an axis-aligned box for simplicity
    def __init__(self, low, high, material, color):
        self.a = low
        self.b = high
        self.material = material
        self.color = color
        
        self.u = ti.Vector([1, 0, 0])
        self.v = ti.Vector([0, 1, 0])
        self.w = ti.Vector([0, 0, 1])
        
        self.center = (low + high) / 2
        
    @ti.func
    def hit(self, ray, t_min=1e-3, t_max=10e8):
        faces = [
            (self.u, ti.Vector([self.b.x, self.a.y, self.a.z]), self.b, ti.Vector([self.b.x, self.b.y, self.a.z])),
            (-self.u, self.a, ti.Vector([self.a.x, self.b.y, self.b.z]), ti.Vector([self.a.x, self.b.y, self.a.z])),
            (self.v, ti.Vector([self.a.x, self.b.y, self.a.z]), self.b, ti.Vector([self.b.x, self.b.y, self.a.z])),
            (-self.v, self.a, ti.Vector([self.b.x, self.a.y, self.b.z]), ti.Vector([self.b.x, self.a.y, self.a.z])),
            (self.w, ti.Vector([self.a.x, self.a.y, self.b.z]), self.b, ti.Vector([self.b.x, self.a.y, self.b.z])),
            (-self.w, self.a, ti.Vector([self.b.x, self.b.y, self.a.z]), ti.Vector([self.b.x, self.a.y, self.a.z]))
        ]
        
        is_hit, root, front_face = False, 10e8, False
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(6)):
            n, p1, p2, p3 = faces[i]
            
            _is_hit = False
            _root = 10e9
            _hit_point = ti.Vector([0.0, 0.0, 0.0])
            if ray.direction.dot(n) != 0:
                _root = ((p1 - ray.origin).dot(n)) / (ray.direction.dot(n))
                _hit_point = ray.at(_root)
                
                _is_hit = 0 < (_hit_point - p3).dot(p2 - p3) < (p2 - p3).dot(p2 - p3) and \
                          0 < (_hit_point - p3).dot(p1 - p3) < (p1 - p3).dot(p1 - p3) and \
                          t_min <= _root <= t_max
            
            _front_face = False
            if _is_hit:
                _hit_point_normal = (_hit_point - self.center).normalized()
                if ray.direction.dot(_hit_point_normal) < 0:
                    _front_face = True
                else:
                    _hit_point_normal = -_hit_point_normal
                if _root < root:
                    is_hit = True
                    root = _root
                    hit_point = _hit_point
                    front_face = _front_face
                    hit_point_normal = _hit_point_normal
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color


@ti.data_oriented
class XYRectangle:
    """
    A rectangle plane aligned to x-axis and y-axis.
    
    defined by z = k
    """
    def __init__(self, x0, x1, y0, y1, k, material, color):
        self.x0 = min(x0, x1)
        self.x1 = max(x0, x1)
        self.y0 = min(y0, y1)
        self.y1 = max(y0, y1)
        self.k = k
        
        self.material = material
        self.color = color
        
    @ti.func
    def hit(self, ray, t_min=1e-3, t_max=10e8):
        t = 0.0
        is_hit = False
        front_face = False
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        outward_normal = ti.Vector([0, 0, 1])
        
        if ray.direction.z == 0.0:
            t = 1.1e9
        else:
            t = (self.k - ray.origin.z) / ray.direction.z
        
        if t_min < t < t_max:
            hit_point = ray.at(t)
            x, y = hit_point.x, hit_point.y
            if self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1:
                is_hit = True
            else:
                is_hit = False
                
            if ray.direction.dot(outward_normal) < 0:
                front_face = True
            else:
                outward_normal = -outward_normal
        return HitRecord(is_hit, t, hit_point, outward_normal, front_face, self.material, self.color)
    
@ti.data_oriented
class XZRectangle:
    """
    A rectangle plane aligned to x-axis and z-axis.
    
    defined by y = k
    """
    def __init__(self, x0, x1, z0, z1, k, material, color):
        self.x0 = min(x0, x1)
        self.x1 = max(x0, x1)
        self.z0 = min(z0, z1)
        self.z1 = max(z0, z1)
        self.k = k
        
        self.material = material
        self.color = color
        
    @ti.func
    def hit(self, ray, t_min=1e-3, t_max=10e8):
        t = 0.0
        is_hit = False
        front_face = False
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        outward_normal = ti.Vector([0, 1, 0])
        
        if ray.direction.y == 0.0:
            t = 1.1e9
        else:
            t = (self.k - ray.origin.y) / ray.direction.y
        
        if t_min < t < t_max:
            hit_point = ray.at(t)
            x, z = hit_point.x, hit_point.z
            if self.x0 <= x <= self.x1 and self.z0 <= z <= self.z1:
                is_hit = True
            else:
                is_hit = False
                
            if ray.direction.dot(outward_normal) < 0:
                front_face = True
            else:
                outward_normal = -outward_normal
        return HitRecord(is_hit, t, hit_point, outward_normal, front_face, self.material, self.color)
    
@ti.data_oriented
class YZRectangle:
    """
    A rectangle plane aligned to y-axis and z-axis.
    
    defined by x = k
    """
    def __init__(self, y0, y1, z0, z1, k, material, color):
        self.y0 = min(y0, y1)
        self.y1 = max(y0, y1)
        self.z0 = min(z0, z1)
        self.z1 = max(z0, z1)
        self.k = k
        
        self.material = material
        self.color = color
        
    @ti.func
    def hit(self, ray, t_min=1e-3, t_max=10e8):
        t = 0.0
        is_hit = False
        front_face = False
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        outward_normal = ti.Vector([1, 0, 0])
        
        if ray.direction.x == 0.0:
            t = 1.1e9
        else:
            t = (self.k - ray.origin.x) / ray.direction.x
        
        if t_min < t < t_max:
            hit_point = ray.at(t)
            y, z = hit_point.y, hit_point.z
            if self.y0 <= y <= self.y1 and self.z0 <= z <= self.z1:
                is_hit = True
            else:
                is_hit = False
                
            if ray.direction.dot(outward_normal) < 0:
                front_face = True
            else:
                outward_normal = -outward_normal
        return HitRecord(is_hit, t, hit_point, outward_normal, front_face, self.material, self.color)
    
@ti.data_oriented
class Box:
    def __init__(self, p0, p1, material, color):
        self.sides = Hittable_list()
        self.material = material
        self.color = color
        
        self.sides.add(XYRectangle(p0.x, p1.x, p0.y, p1.y, p1.z, material, color))
        self.sides.add(XYRectangle(p0.x, p1.x, p0.y, p1.y, p0.z, material, color))
        self.sides.add(XZRectangle(p0.x, p1.x, p0.z, p1.z, p1.y, material, color))
        self.sides.add(XZRectangle(p0.x, p1.x, p0.z, p1.z, p0.y, material, color))
        self.sides.add(YZRectangle(p0.y, p1.y, p0.z, p1.z, p1.x, material, color))
        self.sides.add(YZRectangle(p0.y, p1.y, p0.z, p1.z, p0.x, material, color))
        
    @ti.func
    def hit(self, ray, t_min=1e-3, t_max=10e8):
        return self.sides.hit(ray, t_min, t_max) 

@ti.data_oriented
class RotatedY:
    """rotate an object about the y-aixs
    """
    def __init__(self, box, angle):
        self.box = box
        self.degree = angle / 180.0 * PI
        
    
    @ti.func
    def hit(self, ray, t_min=1e-3, t_max=10e8):
        sin_theta = ti.sin(self.degree)
        cos_theta = ti.cos(self.degree)
        
        r_origin = ray.origin
        r_direction = ray.direction
        
        r_origin.x = cos_theta * ray.origin.x - sin_theta * ray.origin.z
        r_origin.z = sin_theta * ray.origin.x + cos_theta * ray.origin.z
        
        r_direction.x = cos_theta * ray.direction.x - sin_theta * ray.direction.z
        r_direction.z = sin_theta * ray.direction.x + cos_theta * ray.direction.z
        
        r_ray = Ray(r_origin, r_direction)
        
        res = self.box.hit(r_ray, t_min, t_max)
        
        # p = res.hit_point
        # n = res.hit_point_normal
        
        # p.x = cos_theta * res.hit_point.x + sin_theta * res.hit_point.z
        # p.z = cos_theta * res.hit_point.z - sin_theta * res.hit_point.x
        
        # n.x = cos_theta * res.hit_point_normal.x + sin_theta * res.hit_point_normal.z
        # n.z = cos_theta * res.hit_point_normal.z - sin_theta * res.hit_point_normal.x
        
        # res.hit_point = p
        # res.hit_point_normal = n
        return res
        
        